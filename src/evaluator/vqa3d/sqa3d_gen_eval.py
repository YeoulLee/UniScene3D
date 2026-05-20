"""Generative (text exact-match) evaluator for SQA3D."""

import json
from pathlib import Path

import numpy as np

from data.data_utils import clean_answer
from evaluator.common.build import EVALUATOR_REGISTRY


@EVALUATOR_REGISTRY.register()
class SQA3DGenEval():
    """Exact-match evaluator for generated SQA3D answers, with type breakdown.

    Receives batches that the trainer has already gathered across processes;
    it only accumulates and reduces locally (no internal cross-process gather).
    """

    def __init__(self, cfg, accelerator):
        """Args follow the evaluator call convention (cfg, accelerator)."""
        self.target_metric = "em"
        self.best_result = -np.inf
        self.save = cfg.eval.save
        if self.save:
            self.save_dir = Path(cfg.exp_dir) / "eval_results" / "sqa3d_gen"
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self):
        """Clear running counters for a fresh evaluation pass."""
        self.total = 0
        self.correct = 0
        self.type_total = {t: 0 for t in range(6)}
        self.type_correct = {t: 0 for t in range(6)}
        self.eval_results = []

    def update(self, data_dict):
        """Accumulate one already-gathered batch.

        Expects: 'pred' list[str], 'answer_list' list[list[str]],
        'sqa_type' list[int], optionally 'scan_id' list[str].
        """
        preds = data_dict["pred"]
        gts = data_dict["answer_list"]
        types = data_dict["sqa_type"]
        scan_ids = data_dict.get("scan_id", [None] * len(preds))
        for pred, gt_list, t, scan_id in zip(preds, gts, types, scan_ids):
            t = int(t)
            pred_c = clean_answer(pred)
            hit = any(pred_c == clean_answer(g) for g in gt_list)
            self.total += 1
            self.correct += int(hit)
            if t in self.type_total:
                self.type_total[t] += 1
                self.type_correct[t] += int(hit)
            if self.save:
                self.eval_results.append({
                    "scan_id": scan_id,
                    "pred": pred,
                    "gt": list(gt_list),
                    "correct": bool(hit),
                    "sqa_type": t,
                })

    def record(self, split="val"):
        """Finalize metrics; returns (is_best, results_dict)."""
        em = self.correct / max(self.total, 1)
        results = {"em": em, "target_metric": em, "total": self.total}
        for t in range(6):
            results[f"type{t}_acc"] = self.type_correct[t] / max(self.type_total[t], 1)

        is_best = em > self.best_result
        if is_best:
            self.best_result = em
        results["best_result"] = self.best_result

        if self.save and (is_best or split == "test"):
            with (self.save_dir / f"results_{split}.json").open("w") as f:
                json.dump(self.eval_results, f, indent=2)
        return is_best, results
