#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ==== TEXT-ONLY CONTROL (full fine-tuning) ====
# Same full fine-tuning as sqa3d_qwen3d_full.sh but with use_vision=False, so
# Qwen is trained on the question text alone. Compare its EM against the
# with-vision run to measure the 3D pathway's real contribution.
CONFIG="configs/finetune/sqa3d_qwen3d.yaml"
DS_CONFIG="configs/deepspeed_zero2.json"
NOTE="sqa3d_qwen3d_full_textonly_run1"
EXP_NAME="sqa3d_qwen3d_full_textonly_run1"
GPUS=8
cd "${PROJECT_ROOT}"

# ==== SAFETY ====
set -e
set -o pipefail

# ==== OUTPUT LOGGING ====
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/${EXP_NAME}_${NOTE}_$TIMESTAMP.log"

echo "[INFO] Text-only control, full fine-tuning (DeepSpeed ZeRO-2): $EXP_NAME"
echo "[INFO] Logging to: $LOGFILE"

# ==== OFFLINE HF (corporate network blocks huggingface.co) ====
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# export SCENEPOINT_LOCAL_DIR=/path/to/ScenePoint   # if ScenePoint is not in the HF cache

# ==== LAUNCH (vision disabled) ====
python launch.py --mode accelerate --gpu_per_node "$GPUS" --num_nodes 1 \
    --mixed_precision bf16 \
    --deepspeed "$DS_CONFIG" \
    --config "$CONFIG" \
    note="$NOTE" \
    name="$EXP_NAME" \
    solver.gradient_accumulation_steps=1 \
    model.use_vision=False
