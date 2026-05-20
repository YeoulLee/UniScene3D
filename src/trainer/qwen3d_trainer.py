"""Trainer for generative SQA3D with the UniScene3D + Qwen3.5 model."""

import torch
from tqdm import tqdm

from trainer.build import TRAINER_REGISTRY, BaseTrainer


@TRAINER_REGISTRY.register()
class Qwen3DTrainer(BaseTrainer):
    """Causal-LM loss for training; greedy-decode + text exact-match for eval.

    The model (UniScene3DQwen) computes the loss internally and exposes a
    `generate` method, so no external loss module is used.
    """

    def backward(self, loss):
        """One optimizer + scheduler step."""
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        if self.grad_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        self.scheduler.step()

    def train_step(self, epoch):
        """Run one training epoch."""
        self.model.train()
        loader = self.data_loaders["train"]
        pbar = tqdm(
            range(len(loader)),
            disable=(not self.accelerator.is_main_process),
            desc=f"[Epoch {epoch + 1}/{self.epochs}]",
        )
        for data_dict in loader:
            with self.accelerator.accumulate(self.model):
                loss = self.model(data_dict)["loss"]
                self.backward(loss)
                self.global_step += 1
                self.log({"loss": loss.detach(), "step": self.global_step}, mode="train")
                pbar.update(1)

    @torch.no_grad()
    def eval_step(self, epoch, split="val"):
        """Generate answers on a split and score exact match."""
        self.model.eval()
        loader = self.data_loaders[split]
        model = self.accelerator.unwrap_model(self.model)
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process))

        for data_dict in loader:
            preds = model.generate(data_dict)
            gathered = {
                "pred": self.accelerator.gather_for_metrics(preds),
                "answer_list": self.accelerator.gather_for_metrics(data_dict["answer_list"]),
                "sqa_type": self.accelerator.gather_for_metrics(data_dict["sqa_type"]),
                "scan_id": self.accelerator.gather_for_metrics(data_dict["scan_id"]),
            }
            if self.accelerator.is_main_process:
                self.evaluator.update(gathered)
            pbar.update(1)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            is_best, results = self.evaluator.record(split=split)
            self.log(results, mode=split)
            self.accelerator.print(
                f"[Epoch {epoch + 1}] {split} EM={results['em']:.4f} "
                f"(best {results['best_result']:.4f})"
            )
            self.evaluator.reset()
            return is_best
        return False

    def run(self):
        """Train, evaluate per epoch, and checkpoint."""
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.accelerator.print(f"Trainable parameters: {num_trainable:,}")

        start_epoch = self.exp_tracker.epoch
        self.global_step = start_epoch * len(self.data_loaders["train"])
        for epoch in range(start_epoch, self.epochs):
            self.exp_tracker.step()
            self.train_step(epoch)

            if self.epochs_per_eval and (epoch + 1) % self.epochs_per_eval == 0:
                is_best = self.eval_step(epoch, split="val")
            else:
                is_best = False

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.save("latest.pth")
                if is_best:
                    self.save("best.pth")
                if self.epochs_per_save and (epoch + 1) % self.epochs_per_save == 0:
                    self.save(f"ckpt_{epoch + 1}.pth")

        self.accelerator.end_training()
