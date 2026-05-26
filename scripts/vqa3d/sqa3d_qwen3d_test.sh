#!/bin/bash
# SQA3D test-set evaluation for the UniScene3D + Qwen3.5 model.
# Loads a trained checkpoint (DeepSpeed save_state directory) and runs a
# single generate pass on the SQA3D test split.
#
# Usage:
#   CKPT_PATH=results/<EXP_NAME>/ckpt/best.pth bash scripts/vqa3d/sqa3d_qwen3d_test.sh
# Override GPUs, etc.:
#   GPUS=1 CKPT_PATH=results/<EXP_NAME>/ckpt/best.pth bash scripts/vqa3d/sqa3d_qwen3d_test.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ==== REQUIRED ====
if [[ -z "${CKPT_PATH}" ]]; then
    echo "[ERROR] CKPT_PATH is required. Point it at the DeepSpeed save_state directory, e.g.:"
    echo "        CKPT_PATH=results/qwen3d_lr2e-5_.../ckpt/best.pth bash $0"
    exit 1
fi
if [[ ! -d "${CKPT_PATH}" ]]; then
    echo "[ERROR] CKPT_PATH must be a directory (DeepSpeed save_state folder): ${CKPT_PATH}"
    exit 1
fi

# ==== HYPERPARAMETERS (must match the trained model) ====
VOXEL_SIZE="${VOXEL_SIZE:-0.2}"
NUM_TOKENS="${NUM_TOKENS:-512}"
USE_VISION="${USE_VISION:-True}"
GPUS="${GPUS:-8}"
TAG="${TAG:-test1}"
# True = include all test questions (paper convention). False = restrict to
# questions whose GT answer is in the 706-class candidate set (diagnostic only).
USE_UNANSWER="${USE_UNANSWER:-True}"

CONFIG="configs/finetune/sqa3d_qwen3d.yaml"
DS_CONFIG="configs/deepspeed_zero2.json"

CKPT_TAG=$(basename "$(dirname "$(dirname "${CKPT_PATH}")")")
EXP_NAME="eval_${CKPT_TAG}_${TAG}"
NOTE="$EXP_NAME"
cd "${PROJECT_ROOT}"

# ==== SAFETY ====
set -e
set -o pipefail

# ==== OUTPUT LOGGING ====
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/${EXP_NAME}_$TIMESTAMP.log"

echo "[INFO] Eval experiment: $EXP_NAME"
echo "[INFO] Loading from:    $CKPT_PATH"
echo "[INFO] Logging to:      $LOGFILE"

# ==== OFFLINE HF ====
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ==== LAUNCH ====
python launch.py --mode accelerate --gpu_per_node "$GPUS" --num_nodes 1 \
    --mixed_precision bf16 \
    --deepspeed "$DS_CONFIG" \
    --config "$CONFIG" \
    mode=test \
    ++ckpt_path="$CKPT_PATH" \
    note="$NOTE" \
    name="$EXP_NAME" \
    model.voxel_size="$VOXEL_SIZE" \
    model.num_visual_tokens="$NUM_TOKENS" \
    model.use_vision="$USE_VISION" \
    data.ScanNetSQA3DGen.test.use_unanswer="$USE_UNANSWER" \
    eval.save=True \
    2>&1 | tee "$LOGFILE"
