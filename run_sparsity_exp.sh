#!/bin/bash
# Sparsity experiment: baseline + hybrid (sp 0.4~0.8, prefetch_ratio=1.0)
# Usage: CUDA_VISIBLE_DEVICES=0,2,4,5 bash run_sparsity_exp.sh

MODEL_PATH="models/models/DeepSeek_V2_Lite"
TASKS="hellaswag,gsm8k"
FEWSHOT=0
SEED=2026
OUTPUT_DIR="Sparsity_eval/result"
BATCH_SIZE="auto:4"
DTYPE="bfloat16"

echo "=========================================="
echo "Model:  ${MODEL_PATH}"
echo "Tasks:  ${TASKS}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

# 1) Baseline (no sparsity)
echo ""
echo "[1/6] Running baseline (no sparsity) ..."
python eval_sparsity.py \
    --model_path "${MODEL_PATH}" \
    --tasks "${TASKS}" \
    --num_fewshot ${FEWSHOT} \
    --mode none \
    --sparsity_ratio 0.0 \
    --prefetch_expert_ratio 1.0 \
    --seed ${SEED} \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --dtype "${DTYPE}"

# 2) Hybrid mode: sparsity_ratio = 0.4, 0.5, 0.6, 0.7, 0.8
IDX=2
for SP in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
    echo ""
    echo "[${IDX}/6] Running hybrid sp=${SP} prefetch_ratio=1.0 ..."
    python eval_sparsity.py \
        --model_path "${MODEL_PATH}" \
        --tasks "${TASKS}" \
        --num_fewshot ${FEWSHOT} \
        --mode hybrid \
        --sparsity_ratio ${SP} \
        --prefetch_expert_ratio 1.0 \
        --seed ${SEED} \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --dtype "${DTYPE}"
    IDX=$((IDX + 1))
done

echo ""
echo "=========================================="
echo "All experiments done. Results in ${OUTPUT_DIR}/"
ls -la "${OUTPUT_DIR}"/*.json 2>/dev/null
echo "=========================================="
