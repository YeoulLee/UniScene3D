#!/bin/bash
# Capacity + situation-aware-PE run: agent_pose ON, more visual tokens,
# finer voxel grid. Designed as the next experiment after the EM-0.51 plateau
# (epoch-10 dropped vs epoch-5, use_unanswer=False barely changed) which
# indicated the bottleneck is geometric grounding + visual capacity, not
# training length or distribution tail.
#
# Override anything via env vars, e.g.:
#   USE_AGENT_POSE=False bash scripts/vqa3d/sqa3d_qwen3d_agentpose.sh   # ablate
#   GPUS=4 NUM_TOKENS=768 bash scripts/vqa3d/sqa3d_qwen3d_agentpose.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ==== HYPERPARAMETERS (overridable) ====
LR="${LR:-2e-5}"
PROJ_LR="${PROJ_LR:-1e-3}"
VOXEL_SIZE="${VOXEL_SIZE:-0.1}"        # finer grid so the extra tokens land on distinct voxels
NUM_TOKENS="${NUM_TOKENS:-1024}"       # 2x capacity vs the 512-token baseline
EPOCHS="${EPOCHS:-5}"                  # epoch 10 was overfit on the previous setup
USE_VISION="${USE_VISION:-True}"
USE_AGENT_POSE="${USE_AGENT_POSE:-True}"  # situation pose -> agent-frame 3D PE
GPUS="${GPUS:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"          # safety for the longer sequence; drop to 1 if no OOM
TAG="${TAG:-ap_cap}"

CONFIG="configs/finetune/sqa3d_qwen3d.yaml"
DS_CONFIG="configs/deepspeed_zero2.json"

EXP_NAME="qwen3d_lr${LR}_plr${PROJ_LR}_vox${VOXEL_SIZE}_tok${NUM_TOKENS}_ep${EPOCHS}_vis${USE_VISION}_ap${USE_AGENT_POSE}_${TAG}"
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

# ==== OFFLINE HF ====
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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
    model.use_vision="$USE_VISION" \
    model.use_agent_pose="$USE_AGENT_POSE" \
    2>&1 | tee "$LOGFILE"
