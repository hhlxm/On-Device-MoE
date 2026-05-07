#!/bin/bash
# ==========================================================================
#  One-click experiment submission via srun (non-interactive)
#  GPUs are automatically released when the experiment finishes.
#
#  Usage:
#    bash submit_experiment.sh [model_type] [parallel_mode] [gpu_ids] [sparsity_ratios]
#
#  Examples:
#    bash submit_experiment.sh olmoe dp "0"                  # baseline + full sweep 0.1-0.8
#    bash submit_experiment.sh olmoe dp "0" "0.3"            # only sparsity=0.3
#    bash submit_experiment.sh olmoe dp "0,1" "0.1,0.3,0.5"  # multiple specific ratios
#    bash submit_experiment.sh deepseek mp "0,1,2,3"         # baseline + full sweep
# ==========================================================================

set -euo pipefail

MODEL_TYPE="${1:-olmoe}"
PARALLEL_MODE="${2:-dp}"
GPU_IDS="${3:-0}"
SPARSITY_RATIOS="${4:-}"   # e.g. "0.3" or "0.1,0.3,0.5", empty = full sweep

WORK_DIR="/home/fit/renjuliuji/WORK/lxm/On-Device-MoE"
CONDA_ENV="lxm_eval"
PARTITION="h01"
NUM_GPU=2
JOB_NAME="sparsity_${MODEL_TYPE}_${PARALLEL_MODE}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${WORK_DIR}/logs"
LOG_FILE="${LOG_DIR}/experiment_${MODEL_TYPE}_${PARALLEL_MODE}_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

echo "============================================"
echo "  Submitting experiment via srun"
echo "============================================"
echo "  Partition : ${PARTITION}"
echo "  GPUs      : ${NUM_GPU}"
echo "  Model     : ${MODEL_TYPE}"
echo "  Mode      : ${PARALLEL_MODE}"
echo "  GPU IDs   : ${GPU_IDS}"
echo "  Sparsity  : ${SPARSITY_RATIOS:-full sweep (baseline + 0.1~0.8)}"
echo "  Log       : ${LOG_FILE}"
echo "============================================"

srun --partition=${PARTITION} \
     --gres=gpu:${NUM_GPU} \
     --job-name=${JOB_NAME} \
     --kill-on-bad-exit=1 \
     --output="${LOG_FILE}" \
     bash -c "
         source /home/fit/renjuliuji/WORK/anaconda3/etc/profile.d/conda.sh && \
         conda activate ${CONDA_ENV} && \
         cd ${WORK_DIR} && \
         bash run_sparsity_parallel_examples.sh ${MODEL_TYPE} ${PARALLEL_MODE} ${GPU_IDS} 2 ${SPARSITY_RATIOS}
     "

echo ""
echo "Experiment finished. Log saved to: ${LOG_FILE}"
