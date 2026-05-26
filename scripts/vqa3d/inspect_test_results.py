"""Quick inspection of SQA3DGenEval per-sample outputs.

Usage:
    python scripts/vqa3d/inspect_test_results.py <path/to/results_test.json> [N]

Prints:
  - Overall EM and total
  - Per-type accuracy (sqa_type 0..5)
  - N (default 30) random wrong samples with pred vs gt
  - Count of wrong samples whose pred contains 'think' (case-insensitive)
"""

import json
import random
import re
import sys
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    with open(path) as f:
        data = json.load(f)
    total = len(data)
    wrong = [s for s in data if not s["correct"]]
    correct = total - len(wrong)
    print(f"== {path} ==")
    print(f"EM = {correct / max(total, 1):.4f}  ({correct}/{total})")

    by_type = defaultdict(lambda: [0, 0])  # type -> [correct, total]
    for s in data:
        t = int(s["sqa_type"])
        by_type[t][1] += 1
        if s["correct"]:
            by_type[t][0] += 1
    type_names = {0: "what", 1: "is", 2: "how", 3: "can", 4: "which", 5: "other"}
    print("\nper-type accuracy:")
    for t in sorted(by_type):
        c, n = by_type[t]
        print(f"  t{t} ({type_names.get(t, '?'):5s})  {c/max(n,1):.4f}  ({c}/{n})")

    n_thinking = sum(1 for s in wrong if re.search(r"think", s["pred"], re.I))
    print(f"\nwrong with 'think' in pred: {n_thinking} / {len(wrong)}")

    print(f"\n--- {n_show} random wrong samples ---")
    for s in random.sample(wrong, min(n_show, len(wrong))):
        pred = s["pred"]
        gt = " | ".join(s["gt"])
        print(f"  t{s['sqa_type']}  pred={pred!r:40s}  gt=[{gt}]")


if __name__ == "__main__":
    main()
