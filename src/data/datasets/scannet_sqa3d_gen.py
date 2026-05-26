"""SQA3D dataset variant that yields raw text for generative (LLM) training."""

import random

import torch

from ..build import DATASET_REGISTRY
from ..data_utils import get_sqa_question_type, load_safetensor_from_hf
from .scannet import ScanNetSQA3D


def _flatten_position(pos):
    """SQA3D position is a {x,y,z} dict (three.js export); accept list/tuple too."""
    if isinstance(pos, dict):
        return [pos['x'], pos['y'], pos['z']]
    return list(pos)


def _flatten_rotation(rot):
    """SQA3D rotation is a {_x,_y,_z,_w} dict; accept list/tuple too. Returns (x,y,z,w)."""
    if isinstance(rot, dict):
        # three.js uses underscored keys; some dumps use plain keys.
        if '_x' in rot:
            return [rot['_x'], rot['_y'], rot['_z'], rot['_w']]
        return [rot['x'], rot['y'], rot['z'], rot['w']]
    return list(rot)


@DATASET_REGISTRY.register()
class ScanNetSQA3DGen(ScanNetSQA3D):
    """SQA3D dataset for generative QA.

    Reuses ScanNetSQA3D loading (annotations, questions, scans) but returns the
    situation/question/answer as raw text instead of a 706-class one-hot, so a
    language model can be trained and evaluated generatively.
    """

    def __getitem__(self, index):
        """Return one SQA3D sample in generative (text) format."""
        item = self.lang_data[index]
        item_id = item['question_id']
        scan_id = item['scene_id']

        answer_list = [answer['answer'] for answer in item['answers']]

        if self.split == 'train':
            # Train augments with a randomly chosen alternative situation phrasing.
            situation = random.choice(self.questions_map[scan_id][item_id]['situation'])
        else:
            situation = self.questions_map[scan_id][item_id]['situation'][0]
        question = self.questions_map[scan_id][item_id]['question']

        scene_tensor = load_safetensor_from_hf(
            repo_id="MatchLab/ScenePoint",
            filename=self.scan_data[scan_id]["safetensors_path"],
        )
        point_map = scene_tensor['point_map'].permute(0, 3, 1, 2)
        images = scene_tensor['color_images'].permute(0, 3, 1, 2)

        # SQA3D annotation provides agent pose (position + rotation as quaternion).
        # Both are three.js-style dicts ({"x":..,"y":..,"z":..} and
        # {"_x":..,"_y":..,"_z":..,"_w":..}); flatten to tensors so the model
        # can transform voxel coords into the agent's frame for situation-aware PE.
        agent_position = torch.tensor(
            _flatten_position(item['position']), dtype=torch.float32,
        )  # (3,)
        agent_rotation = torch.tensor(
            _flatten_rotation(item['rotation']), dtype=torch.float32,
        )  # (4,) (x, y, z, w)

        return {
            "situation": situation,                              # str
            "question": question,                                # str
            "answer": answer_list[0] if answer_list else "",     # str, generation target
            "answer_list": answer_list,                          # list[str], all GT for eval EM
            "point_map": point_map,                              # (V, 3, H, W)
            "images": images,                                    # (V, 3, H, W)
            "agent_position": agent_position,                    # (3,)
            "agent_rotation": agent_rotation,                    # (4,)
            "sqa_type": get_sqa_question_type(question),         # int 0-5
            "scan_id": scan_id,                                  # str
            "data_idx": item_id,                                 # question id
        }
