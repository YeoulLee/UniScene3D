"""UniScene3D + Qwen3.5 generative 3D scene QA model.

FG-CLIP scene encoder (frozen) -> per-patch voxel tokens + additive 3D PE ->
projector -> Qwen3.5 language model. The Qwen vision tower is bypassed: visual
tokens are concatenated with text token embeddings and fed as `inputs_embeds`.
"""

import re
from pathlib import Path

import torch
from torch.amp import autocast
from transformers import AutoModelForImageTextToText, AutoTokenizer

from common.misc import build_fgclip_model_from_local_code_with_hf_weights
from model.build import MODEL_REGISTRY, BaseModel
from modules.coord_pe import Sinusoidal3DPositionEncoding, extract_patch_coords
from modules.qwen3d_projector import Qwen3DProjector
from modules.voxel_pooling import VoxelPooling
from optim.utils import no_decay_param_group

_FGCLIP_VIS_DIM = 768   # FG-CLIP vision last_hidden_state hidden size
_FGCLIP_PATCH = 16      # FG-CLIP vision patch size


@MODEL_REGISTRY.register()
class UniScene3DQwen(BaseModel):
    """Generative SQA3D model bridging the UniScene3D encoder to Qwen3.5."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        m = cfg.model

        # --- frozen FG-CLIP scene encoder ---
        fgclip_root = str(Path(__file__).resolve().parents[1] / "fg-clip")
        self.pm_encoder = build_fgclip_model_from_local_code_with_hf_weights(fgclip_root)
        for p in self.pm_encoder.parameters():
            p.requires_grad = False
        self.pm_encoder.eval()
        self.patch_size = _FGCLIP_PATCH

        # --- Qwen3.5 language model (vision tower loaded but unused) ---
        qwen_path = m.qwen_model_path
        self.qwen = AutoModelForImageTextToText.from_pretrained(
            qwen_path, dtype=torch.bfloat16, local_files_only=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        llm_dim = self.qwen.config.text_config.hidden_size  # 2560

        # --- Qwen tuning mode: "full" | "lora" | "frozen" ---
        self.qwen_tuning = m.get("qwen_tuning", "full")
        if self.qwen_tuning not in ("full", "lora", "frozen"):
            raise ValueError(f"unknown qwen_tuning '{self.qwen_tuning}'")
        if self.qwen_tuning in ("frozen", "lora"):
            for p in self.qwen.parameters():
                p.requires_grad = False
        if self.qwen_tuning == "lora":
            from peft import LoraConfig, get_peft_model
            lora = m.get("lora", {})
            default_targets = [
                "q_proj", "k_proj", "v_proj", "o_proj",                  # full attention
                "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",  # linear attn
                "gate_proj", "up_proj", "down_proj",                     # MLP
            ]
            self.qwen = get_peft_model(self.qwen, LoraConfig(
                r=lora.get("r", 16),
                lora_alpha=lora.get("alpha", 32),
                lora_dropout=lora.get("dropout", 0.05),
                target_modules=list(lora.get("target_modules", default_targets)),
                task_type="CAUSAL_LM",
            ))
        # Gradient checkpointing keeps Qwen training within GPU memory.
        if self.qwen_tuning != "frozen" and m.get("gradient_checkpointing", True):
            self.qwen.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # --- 3D scene token modules (projector is trainable) ---
        self.voxel_pool = VoxelPooling(
            voxel_size=m.get("voxel_size", 0.2),
            max_tokens=m.get("num_visual_tokens", 512),
        )
        self.coord_pe = Sinusoidal3DPositionEncoding(dim=_FGCLIP_VIS_DIM)
        self.projector = Qwen3DProjector(in_dim=_FGCLIP_VIS_DIM, out_dim=llm_dim)

        self.max_new_tokens = m.get("max_new_tokens", 16)
        # use_vision=False feeds text only (no visual tokens) -> text-only baseline.
        self.use_vision = m.get("use_vision", True)

    def train(self, mode=True):
        """Keep the frozen FG-CLIP encoder in eval mode regardless of mode."""
        super().train(mode)
        self.pm_encoder.eval()
        return self

    # ---- scene encoding -------------------------------------------------
    @torch.no_grad()
    def _encode_patches(self, images, point_map):
        """FG-CLIP on RGB+pointmap -> patch tokens (B, V, P, vis_dim)."""
        B, V = images.shape[:2]
        H, W = images.shape[-2:]
        imgs = images.reshape(B * V, *images.shape[2:]).float()
        pm = point_map.reshape(B * V, *point_map.shape[2:]).float()
        color_pm = torch.cat([imgs, pm], dim=1)  # (B*V, 6, H, W)
        with autocast("cuda", dtype=torch.bfloat16):
            last_hidden, _ = self.pm_encoder.get_image_features(color_pm)
        P = (H // self.patch_size) * (W // self.patch_size)
        if last_hidden.shape[1] == P + 1:
            last_hidden = last_hidden[:, 1:, :]  # drop CLS token
        elif last_hidden.shape[1] != P:
            raise RuntimeError(
                f"FG-CLIP returned {last_hidden.shape[1]} tokens, expected {P} or {P + 1}."
            )
        return last_hidden.reshape(B, V, P, -1)

    def _scene_tokens(self, data_dict):
        """Produce projected visual tokens (B, N, llm_dim) and mask (B, N)."""
        patch_tokens = self._encode_patches(data_dict["images"], data_dict["point_map"])
        coords, valid = extract_patch_coords(data_dict["point_map"], self.patch_size)
        voxel_feat, voxel_coord, voxel_mask = self.voxel_pool(patch_tokens, coords, valid)
        with autocast("cuda", dtype=torch.bfloat16):
            voxel_feat = voxel_feat + self.coord_pe(voxel_coord)  # (B, N, vis_dim)
            visual_embeds = self.projector(voxel_feat)            # (B, N, llm_dim)
        return visual_embeds, voxel_mask

    def _build_prompt(self, situation, question):
        """Plain text prompt; the answer span follows 'Answer:'."""
        return f"Situation: {situation}\nQuestion: {question}\nAnswer:"

    # ---- training forward ----------------------------------------------
    def forward(self, data_dict):
        """Training forward; returns {'loss': causal-LM loss on the answer span}."""
        embed = self.qwen.get_input_embeddings()
        device = embed.weight.device
        B = len(data_dict["question"])
        pad_id, eos_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id

        seqs, label_seqs = [], []
        for i in range(B):
            p_ids = self.tokenizer(
                self._build_prompt(data_dict["situation"][i], data_dict["question"][i]),
                add_special_tokens=True,
            ).input_ids
            a_ids = self.tokenizer(
                " " + data_dict["answer"][i], add_special_tokens=False,
            ).input_ids + [eos_id]
            seqs.append(p_ids + a_ids)
            label_seqs.append([-100] * len(p_ids) + a_ids)

        T = max(len(s) for s in seqs)
        text_ids, text_mask, text_labels = [], [], []
        for seq, lab in zip(seqs, label_seqs):
            n_pad = T - len(seq)
            text_ids.append(seq + [pad_id] * n_pad)         # right-pad for training
            text_mask.append([1] * len(seq) + [0] * n_pad)
            text_labels.append(lab + [-100] * n_pad)
        text_ids = torch.tensor(text_ids, device=device)
        attention_mask = torch.tensor(text_mask, device=device)
        labels = torch.tensor(text_labels, device=device)
        inputs_embeds = embed(text_ids)

        if self.use_vision:
            visual_embeds, voxel_mask = self._scene_tokens(data_dict)
            N = voxel_mask.shape[1]
            visual_embeds = visual_embeds.to(embed.weight.dtype)
            inputs_embeds = torch.cat([visual_embeds, inputs_embeds], dim=1)
            attention_mask = torch.cat([voxel_mask.long(), attention_mask], dim=1)
            labels = torch.cat([
                torch.full((B, N), -100, dtype=torch.long, device=device), labels,
            ], dim=1)

        out = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {"loss": out.loss}

    # ---- generative eval -----------------------------------------------
    @torch.no_grad()
    def generate(self, data_dict):
        """Greedy-generate the answer text for each sample; returns list[str]."""
        embed = self.qwen.get_input_embeddings()
        device = embed.weight.device
        B = len(data_dict["question"])
        pad_id = self.tokenizer.pad_token_id

        prompts = [
            self.tokenizer(
                self._build_prompt(data_dict["situation"][i], data_dict["question"][i]),
                add_special_tokens=True,
            ).input_ids
            for i in range(B)
        ]
        T = max(len(p) for p in prompts)
        text_ids, text_mask = [], []
        for p_ids in prompts:
            n_pad = T - len(p_ids)
            text_ids.append([pad_id] * n_pad + p_ids)        # left-pad for generation
            text_mask.append([0] * n_pad + [1] * len(p_ids))
        text_ids = torch.tensor(text_ids, device=device)
        attention_mask = torch.tensor(text_mask, device=device)
        inputs_embeds = embed(text_ids)

        if self.use_vision:
            visual_embeds, voxel_mask = self._scene_tokens(data_dict)
            visual_embeds = visual_embeds.to(embed.weight.dtype)
            inputs_embeds = torch.cat([visual_embeds, inputs_embeds], dim=1)
            attention_mask = torch.cat([voxel_mask.long(), attention_mask], dim=1)

        gen_ids = self.qwen.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return [self._postprocess_answer(t) for t in texts]

    @staticmethod
    def _postprocess_answer(text):
        """Strip <think> blocks, leaked prompt prefix, and post-answer fluff.

        Qwen3 may emit a <think>...</think> reasoning block before the answer,
        and even with skip_special_tokens=True the literal tags can survive.
        We also drop a leaked 'Answer:' prefix and keep only the first line.
        """
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"</?think>", "", text)
        text = text.strip()
        text = text.split("\n", 1)[0]
        text = re.sub(r"^\s*answer\s*:\s*", "", text, flags=re.IGNORECASE)
        return text.strip()

    def get_opt_params(self):
        """Trainable param groups: projector (own lr) + trainable Qwen params."""
        proj_lr = self.cfg.model.get("projector_lr", self.cfg.solver.lr)
        groups = no_decay_param_group(self.projector.named_parameters(), proj_lr)
        qwen_named = [(n, p) for n, p in self.qwen.named_parameters() if p.requires_grad]
        if qwen_named:
            groups += no_decay_param_group(qwen_named, self.cfg.solver.lr)
        return groups
