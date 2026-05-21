#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ==== USER SETTINGS ====
CONFIG="configs/finetune/sqa3d_qwen3d.yaml"
NOTE="sqa3d_qwen3d_stage1_gpu8_run1"
EXP_NAME="sqa3d_qwen3d_stage1_gpu8_run1"
GPUS=8
# Effective global batch = batchsize(4) x GPUS(8) x grad_accum(1) = 32.
# lr is sqrt-scaled from the 1e-3 (batch-16) baseline: 1e-3 * sqrt(2) ~= 1.4e-3.
GRAD_ACCUM=1
LR=1.4e-3
cd "${PROJECT_ROOT}"

# ==== SAFETY ====
set -e
set -o pipefail

# ==== OUTPUT LOGGING ====
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/${EXP_NAME}_${NOTE}_$TIMESTAMP.log"

echo "[INFO] Starting training: $EXP_NAME ($NOTE)"
echo "[INFO] Logging to: $LOGFILE"

# ==== OFFLINE HF (corporate network blocks huggingface.co) ====
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# export SCENEPOINT_LOCAL_DIR=/path/to/ScenePoint   # if ScenePoint is not in the HF cache

# ==== LAUNCH ====
python launch.py --mode accelerate --gpu_per_node "$GPUS" --num_nodes 1 \
    --mixed_precision bf16 \
    --config "$CONFIG" \
    note="$NOTE" \
    name="$EXP_NAME" \
    solver.gradient_accumulation_steps="$GRAD_ACCUM" \
    solver.lr="$LR"
