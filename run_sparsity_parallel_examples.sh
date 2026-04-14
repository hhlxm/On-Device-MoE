#!/bin/bash
# ==========================================================================
#  Sparsity evaluation — parallel launch examples
#
#  Three modes:
#    1) Model Parallel (MP)  — model sharded across N GPUs, single process
#    2) Data Parallel  (DP)  — model copied to N GPUs, N processes
#    3) MP + DP combined     — model sharded across M GPUs, N/M processes
#
#  Usage:
#    bash run_sparsity_parallel_examples.sh <mp|dp|mp_dp> [GPUS]
#
#  Examples:
#    bash run_sparsity_parallel_examples.sh mp   "4,5,6,7"
#    bash run_sparsity_parallel_examples.sh dp   "4,5,6,7"
#    bash run_sparsity_parallel_examples.sh mp_dp "0,1,2,3,4,5,6,7"
# ==========================================================================

set -euo pipefail

# ---------------------- Common config ----------------------
MODEL_PATH="models/DeepSeek_V2_Lite"
TASKS="mmlu"
FEWSHOT=5
SEED=2026
OUTPUT_DIR="Sparsity_eval/result"
BATCH_SIZE="auto:4"
DTYPE="bfloat16"

MODE="hybrid"
SPARSITY_RATIO=0.5
PREFETCH_EXPERT_RATIO=1.0

# ---------------------- Parse arguments ----------------------
PARALLEL_MODE="${1:-mp}"
GPU_IDS="${2:-0,1,2,3}"

# Count GPUs
IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "=========================================="
echo "Parallel mode : ${PARALLEL_MODE}"
echo "GPUs          : ${GPU_IDS} (${NUM_GPUS} total)"
echo "Model         : ${MODEL_PATH}"
echo "Tasks         : ${TASKS}"
echo "Sparsity      : mode=${MODE}, ratio=${SPARSITY_RATIO}"
echo "=========================================="


# ==============================================================
#  1) Model Parallel — single process, model sharded across GPUs
# ==============================================================
#  When to use:
#    - Model is too large for a single GPU
#    - You want to use all GPUs for one model copy
#  How it works:
#    - gpus_per_model = NUM_GPUS  → device_map="auto" shards layers
#    - Single python process, no accelerate launch needed
#    - All visible GPUs are used by one model via device_map="auto"
#
#  Example topology (4 GPUs):
#    GPU 4: layers 0-6    ┐
#    GPU 5: layers 7-13   │ 1 model copy
#    GPU 6: layers 14-20  │
#    GPU 7: layers 21-27  ┘
# ==============================================================
run_model_parallel() {
    echo ""
    echo "[Model Parallel] ${NUM_GPUS} GPUs -> 1 model copy sharded"
    echo ""

    CUDA_VISIBLE_DEVICES=${GPU_IDS} python eval_sparsity.py \
        --model_path "${MODEL_PATH}" \
        --tasks "${TASKS}" \
        --num_fewshot ${FEWSHOT} \
        --mode "${MODE}" \
        --sparsity_ratio ${SPARSITY_RATIO} \
        --prefetch_expert_ratio ${PREFETCH_EXPERT_RATIO} \
        --seed ${SEED} \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --dtype "${DTYPE}" \
        --gpus_per_model ${NUM_GPUS}
}


# ==============================================================
#  2) Data Parallel — N processes, each with 1 GPU, 1 model copy
# ==============================================================
#  When to use:
#    - Model fits on a single GPU
#    - You want to speed up evaluation by processing data in parallel
#  How it works:
#    - gpus_per_model = 1  → each process loads a full model on 1 GPU
#    - accelerate launch spawns NUM_GPUS processes
#    - lm_eval splits the dataset across processes automatically
#
#  Example topology (4 GPUs):
#    GPU 4: model copy 0  → processes samples 0,4,8,...
#    GPU 5: model copy 1  → processes samples 1,5,9,...
#    GPU 6: model copy 2  → processes samples 2,6,10,...
#    GPU 7: model copy 3  → processes samples 3,7,11,...
# ==============================================================
run_data_parallel() {
    local NUM_PROCS=${NUM_GPUS}

    echo ""
    echo "[Data Parallel] ${NUM_GPUS} GPUs -> ${NUM_PROCS} model copies (1 GPU each)"
    echo ""

    CUDA_VISIBLE_DEVICES=${GPU_IDS} \
    accelerate launch --num_processes ${NUM_PROCS} \
        eval_sparsity.py \
        --model_path "${MODEL_PATH}" \
        --tasks "${TASKS}" \
        --num_fewshot ${FEWSHOT} \
        --mode "${MODE}" \
        --sparsity_ratio ${SPARSITY_RATIO} \
        --prefetch_expert_ratio ${PREFETCH_EXPERT_RATIO} \
        --seed ${SEED} \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --dtype "${DTYPE}" \
        --gpus_per_model 1
}


# ==============================================================
#  3) Model Parallel + Data Parallel — hybrid
# ==============================================================
#  When to use:
#    - Model is too large for 1 GPU (needs M GPUs)
#    - You have more than M GPUs and want data parallelism too
#  How it works:
#    - gpus_per_model = GPUS_PER_MODEL (e.g. 2)
#    - num_processes  = NUM_GPUS / GPUS_PER_MODEL (e.g. 8/2=4)
#    - Each process gets its own GPU slice via CUDA_VISIBLE_DEVICES
#    - Within each slice, device_map="auto" shards the model
#
#  Example topology (8 GPUs, gpus_per_model=2):
#    GPU 0,1: model copy 0  → process 0, samples 0,4,8,...
#    GPU 2,3: model copy 1  → process 1, samples 1,5,9,...
#    GPU 4,5: model copy 2  → process 2, samples 2,6,10,...
#    GPU 6,7: model copy 3  → process 3, samples 3,7,11,...
#
#  Example topology (4 GPUs, gpus_per_model=2):
#    GPU 4,5: model copy 0  → process 0, samples 0,2,4,...
#    GPU 6,7: model copy 1  → process 1, samples 1,3,5,...
# ==============================================================
GPUS_PER_MODEL=${3:-2}    # 3rd argument, default 2

run_mp_dp() {
    if (( NUM_GPUS % GPUS_PER_MODEL != 0 )); then
        echo "ERROR: NUM_GPUS (${NUM_GPUS}) must be divisible by GPUS_PER_MODEL (${GPUS_PER_MODEL})"
        exit 1
    fi

    local NUM_PROCS=$(( NUM_GPUS / GPUS_PER_MODEL ))

    echo ""
    echo "[MP+DP] ${NUM_GPUS} GPUs, ${GPUS_PER_MODEL} per model -> ${NUM_PROCS} model copies"
    echo ""

    CUDA_VISIBLE_DEVICES=${GPU_IDS} \
    accelerate launch --num_processes ${NUM_PROCS} \
        eval_sparsity.py \
        --model_path "${MODEL_PATH}" \
        --tasks "${TASKS}" \
        --num_fewshot ${FEWSHOT} \
        --mode "${MODE}" \
        --sparsity_ratio ${SPARSITY_RATIO} \
        --prefetch_expert_ratio ${PREFETCH_EXPERT_RATIO} \
        --seed ${SEED} \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --dtype "${DTYPE}" \
        --gpus_per_model ${GPUS_PER_MODEL}
}


# ---------------------- Dispatch ----------------------
case "${PARALLEL_MODE}" in
    mp)
        run_model_parallel
        ;;
    dp)
        run_data_parallel
        ;;
    mp_dp)
        run_mp_dp
        ;;
    *)
        echo "Unknown mode: ${PARALLEL_MODE}"
        echo "Usage: $0 <mp|dp|mp_dp> [GPU_IDS] [GPUS_PER_MODEL]"
        echo ""
        echo "  mp    — Model Parallel:  model sharded across all GPUs"
        echo "  dp    — Data Parallel:   N model copies, 1 GPU each"
        echo "  mp_dp — MP + DP:         model sharded across M GPUs, N/M copies"
        echo ""
        echo "Examples:"
        echo "  $0 mp    4,5,6,7          # 4 GPU model parallel"
        echo "  $0 dp    4,5,6,7          # 4-way data parallel"
        echo "  $0 mp_dp 0,1,2,3,4,5,6,7 2  # 2 GPU/model, 4 copies"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Done. Results in ${OUTPUT_DIR}/"
ls -la "${OUTPUT_DIR}"/*.json 2>/dev/null || true
echo "=========================================="

