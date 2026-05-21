#!/bin/bash
# Parameterised experiment runner for the UniScene3D + Qwen3.5 full-FT pipeline.
# Override any hyperparameter via an environment variable, e.g.:
#   LR=1e-5 VOXEL_SIZE=0.1 bash scripts/vqa3d/sqa3d_qwen3d_exp.sh
# Sweep example:
#   for lr in 1e-5 2e-5 5e-5; do LR=$lr TAG=lrsweep bash scripts/vqa3d/sqa3d_qwen3d_exp.sh; done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ==== HYPERPARAMETERS (override via env vars) ====
LR="${LR:-2e-5}"                  # Qwen full-FT learning rate
PROJ_LR="${PROJ_LR:-1e-3}"        # projector learning rate
VOXEL_SIZE="${VOXEL_SIZE:-0.2}"   # 3D voxel size in metres
NUM_TOKENS="${NUM_TOKENS:-512}"   # visual token budget
EPOCHS="${EPOCHS:-5}"
USE_VISION="${USE_VISION:-True}"  # False = text-only control
GPUS="${GPUS:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
TAG="${TAG:-run1}"

CONFIG="configs/finetune/sqa3d_qwen3d.yaml"
DS_CONFIG="configs/deepspeed_zero2.json"

# Auto-named so each hyperparameter combo lands in its own results folder.
EXP_NAME="qwen3d_lr${LR}_plr${PROJ_LR}_vox${VOXEL_SIZE}_tok${NUM_TOKENS}_ep${EPOCHS}_vis${USE_VISION}_${TAG}"
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

echo "[INFO] Experiment: $EXP_NAME"
echo "[INFO] Logging to: $LOGFILE"

# ==== OFFLINE HF (corporate network blocks huggingface.co) ====
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# export SCENEPOINT_LOCAL_DIR=/path/to/ScenePoint   # if ScenePoint is not in the HF cache

# ==== LAUNCH ====
python launch.py --mode accelerate --gpu_per_node "$GPUS" --num_nodes 1 \
    --mixed_precision bf16 \
    --deepspeed "$DS_CONFIG" \
    --config "$CONFIG" \
    note="$NOTE" \
    name="$EXP_NAME" \
    solver.lr="$LR" \
    model.projector_lr="$PROJ_LR" \
    model.voxel_size="$VOXEL_SIZE" \
    model.num_visual_tokens="$NUM_TOKENS" \
    solver.epochs="$EPOCHS" \
    solver.gradient_accumulation_steps="$GRAD_ACCUM" \
    model.use_vision="$USE_VISION"
