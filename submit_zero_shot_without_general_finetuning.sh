#!/bin/bash
# ==========================================================================
#  One-click zero-shot sparse evaluation submission via srun (non-interactive)
#  GPUs are automatically released when the experiment finishes.
#
#  Usage:
#    bash submit_zero_shot_without_general_finetuning.sh [cuda] [model_name] [sparsity_list] [extra_args...]
#
#  Example:
#    bash submit_zero_shot_without_general_finetuning.sh
#    bash submit_zero_shot_without_general_finetuning.sh 4 sparse_Deepseek "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8"
#    bash submit_zero_shot_without_general_finetuning.sh 4 sparse_Deepseek "0.5,0.7" --tasks gsm8k --limit 10
#
#  The default submitted command is equivalent to:
#    bash scripts/zero_shot_evaluation_without_general_finetuning.sh \
#      --cuda 4 \
#      --model-name sparse_Deepseek \
#      --sparsity-list 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8
# ==========================================================================

set -euo pipefail

CUDA_DEVICES="${1:-4}"
MODEL_NAME="${2:-sparse_Deepseek}"
SPARSITY_LIST="${3:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8}"
EXTRA_ARGS=("${@:4}")

WORK_DIR="${WORK_DIR:-/home/fit/renjuliuji/WORK/lxm/On-Device-MoE}"
CONDA_ENV="${CONDA_ENV:-lxm_eval}"
PARTITION="${PARTITION:-h01}"

IFS=',' read -r -a CUDA_ID_LIST <<< "${CUDA_DEVICES}"
NUM_GPU="${NUM_GPU:-${#CUDA_ID_LIST[@]}}"

JOB_NAME="zero_shot_${MODEL_NAME}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${WORK_DIR}/logs"
LOG_FILE="${LOG_DIR}/zero_shot_without_general_finetuning_${MODEL_NAME}_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

printf -v CUDA_DEVICES_Q '%q' "${CUDA_DEVICES}"
printf -v MODEL_NAME_Q '%q' "${MODEL_NAME}"
printf -v SPARSITY_LIST_Q '%q' "${SPARSITY_LIST}"
EXTRA_ARGS_Q=""
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  printf -v EXTRA_ARGS_Q ' %q' "${EXTRA_ARGS[@]}"
fi

echo "=============================================================="
echo "  Submitting zero-shot sparse evaluation via srun"
echo "=============================================================="
echo "  Partition : ${PARTITION}"
echo "  GPUs      : ${NUM_GPU}"
echo "  CUDA      : ${CUDA_DEVICES}"
echo "  Model     : ${MODEL_NAME}"
echo "  Sparsity  : ${SPARSITY_LIST}"
echo "  Extra args:${EXTRA_ARGS_Q:- none}"
echo "  Log       : ${LOG_FILE}"
echo "=============================================================="

srun --partition="${PARTITION}" \
     --gres="gpu:${NUM_GPU}" \
     --job-name="${JOB_NAME}" \
     --kill-on-bad-exit=1 \
     --output="${LOG_FILE}" \
     bash -c "
         source /home/fit/renjuliuji/WORK/anaconda3/etc/profile.d/conda.sh && \
         conda activate ${CONDA_ENV} && \
         cd ${WORK_DIR}/CATS && \
         bash scripts/zero_shot_evaluation_without_general_finetuning.sh \
           --cuda ${CUDA_DEVICES_Q} \
           --model-name ${MODEL_NAME_Q} \
           --sparsity-list ${SPARSITY_LIST_Q}${EXTRA_ARGS_Q}
     "

echo ""
echo "Experiment finished. Log saved to: ${LOG_FILE}"
