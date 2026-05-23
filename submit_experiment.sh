#!/bin/bash
# ==========================================================================
#  One-click experiment submission via srun (non-interactive)
#  GPUs are automatically released when the experiment finishes.
#
#  Usage:
#    bash submit_experiment.sh [model_type] [parallel_mode] [gpu_ids] [sparsity_ratios] [extra args...]
#
#  Examples:
#    bash submit_experiment.sh olmoe dp "0"                  # baseline + full sweep 0.1-0.8
#    bash submit_experiment.sh olmoe dp "0" "0.3"            # only sparsity=0.3
#    bash submit_experiment.sh olmoe dp "0,1" "0.1,0.3,0.5"  # multiple specific ratios
#    bash submit_experiment.sh olmoe dp "0" --baseline-only
#    bash submit_experiment.sh deepseek mp "0,1,2,3"         # baseline + full sweep
# ==========================================================================

set -euo pipefail

MODEL_TYPE="${1:-olmoe}"
PARALLEL_MODE="${2:-dp}"
GPU_IDS="${3:-0}"
OPTIONAL_ARGS=("${@:4}")
SPARSITY_ARGS=()
EXTRA_ARGS=()

for arg in "${OPTIONAL_ARGS[@]}"; do
    case "${arg}" in
        --gate|--baseline-only)
            EXTRA_ARGS+=("${arg}")
            ;;
        *)
            SPARSITY_ARGS+=("${arg}")
            ;;
    esac
done

SPARSITY_RATIOS="${SPARSITY_ARGS[0]:-}"   # e.g. "0.3" or "0.1,0.3,0.5", empty = full sweep

if (( ${#SPARSITY_ARGS[@]} > 1 )); then
    echo "Unexpected extra positional argument(s): ${SPARSITY_ARGS[*]:1}"
    echo "Usage: $0 [model_type] [mp|dp|mp_dp] [gpu_ids] [sparsity_ratios] [--gate] [--baseline-only]"
    exit 1
fi

WORK_DIR="/home/fit/renjuliuji/WORK/lxm/On-Device-MoE"
CONDA_ENV="lxm_eval"
PARTITION="h01"
JOB_NAME="sparsity_${MODEL_TYPE}_${PARALLEL_MODE}"

IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPU=${#GPU_ARRAY[@]}
if (( NUM_GPU < 1 )); then
    echo "GPU IDs must not be empty."
    exit 1
fi

GPUS_PER_MODEL="${GPUS_PER_MODEL:-2}"

RUN_ARGS=("${MODEL_TYPE}" "${PARALLEL_MODE}" "${GPU_IDS}")
case "${PARALLEL_MODE}" in
    mp|dp)
        # Keep the runner's optional 4th positional argument empty for mp/dp;
        # otherwise it is interpreted as LIMIT.
        RUN_ARGS+=("" "${SPARSITY_RATIOS}")
        ;;
    mp_dp)
        RUN_ARGS+=("${GPUS_PER_MODEL}" "${SPARSITY_RATIOS}")
        ;;
    *)
        echo "Unknown parallel_mode: ${PARALLEL_MODE}"
        echo "Usage: $0 [model_type] [mp|dp|mp_dp] [gpu_ids] [sparsity_ratios] [extra args...]"
        exit 1
        ;;
esac
RUN_ARGS+=("${EXTRA_ARGS[@]}")
printf -v RUN_ARGS_QUOTED ' %q' "${RUN_ARGS[@]}"

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
if [[ "${PARALLEL_MODE}" == "mp_dp" ]]; then
    echo "  GPUs/model: ${GPUS_PER_MODEL}"
fi
echo "  Sparsity  : ${SPARSITY_RATIOS:-full sweep (baseline + 0.1~0.8)}"
echo "  Extra args: ${EXTRA_ARGS[*]:-none}"
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
         bash run_sparsity_parallel_examples.sh${RUN_ARGS_QUOTED}
     "

echo ""
echo "Experiment finished. Log saved to: ${LOG_FILE}"
