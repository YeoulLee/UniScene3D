#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ==== TEXT-ONLY BASELINE ====
# Qwen answers from the question text alone (no visual tokens), eval-only
# (mode=test, no training) -> measures the language-prior floor. Compare this
# EM against the with-vision run to see whether the 3D pathway contributes.
CONFIG="configs/finetune/sqa3d_qwen3d.yaml"
NOTE="sqa3d_qwen3d_textonly"
EXP_NAME="sqa3d_qwen3d_textonly"
GPUS=1
cd "${PROJECT_ROOT}"

# ==== SAFETY ====
set -e
set -o pipefail

# ==== OUTPUT LOGGING ====
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/${EXP_NAME}_${NOTE}_$TIMESTAMP.log"

echo "[INFO] Text-only baseline (eval-only): $EXP_NAME"
echo "[INFO] Logging to: $LOGFILE"

# ==== OFFLINE HF (corporate network blocks huggingface.co) ====
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# export SCENEPOINT_LOCAL_DIR=/path/to/ScenePoint   # if ScenePoint is not in the HF cache

# ==== LAUNCH (eval-only, vision disabled) ====
python launch.py --mode accelerate --gpu_per_node "$GPUS" --num_nodes 1 \
    --mixed_precision bf16 \
    --config "$CONFIG" \
    note="$NOTE" \
    name="$EXP_NAME" \
    mode=test \
    model.use_vision=False
