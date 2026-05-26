#!/bin/bash
# Test-set evaluation for checkpoints trained with sqa3d_qwen3d_agentpose.sh.
# Defaults mirror the training script (NUM_TOKENS=1024, VOXEL_SIZE=0.1,
# USE_AGENT_POSE=True) -- these MUST match the trained model or load_state_dict
# will produce shape mismatches.
#
# Usage:
#   CKPT_PATH=results/<EXP>/ckpt/best.pth bash scripts/vqa3d/sqa3d_qwen3d_agentpose_test.sh
# Override anything:
#   GPUS=4 USE_UNANSWER=False CKPT_PATH=... bash scripts/vqa3d/sqa3d_qwen3d_agentpose_test.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ==== REQUIRED ====
if [[ -z "${CKPT_PATH}" ]]; then
    echo "[ERROR] CKPT_PATH is required. Point it at the DeepSpeed save_state directory, e.g.:"
    echo "        CKPT_PATH=results/qwen3d_..._apTrue_.../ckpt/best.pth bash $0"
    exit 1
fi
if [[ ! -d "${CKPT_PATH}" ]]; then
    echo "[ERROR] CKPT_PATH must be a directory (DeepSpeed save_state folder): ${CKPT_PATH}"
    exit 1
fi

# ==== HYPERPARAMETERS (must match the trained model) ====
VOXEL_SIZE="${VOXEL_SIZE:-0.1}"
NUM_TOKENS="${NUM_TOKENS:-1024}"
USE_VISION="${USE_VISION:-True}"
USE_AGENT_POSE="${USE_AGENT_POSE:-True}"
GPUS="${GPUS:-8}"
TAG="${TAG:-ap_test}"
# True = full test split (paper convention). False = answerable subset (diagnostic).
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
    model.use_agent_pose="$USE_AGENT_POSE" \
    data.ScanNetSQA3DGen.test.use_unanswer="$USE_UNANSWER" \
    eval.save=True \
    2>&1 | tee "$LOGFILE"
