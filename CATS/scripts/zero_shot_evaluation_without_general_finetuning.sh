#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cats_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${cats_dir}/.." && pwd)"

export PYTHONPATH="${repo_root}:${cats_dir}:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1

model_name="sparse_OLMoE"
base_model_repo_id=""
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-7}"
sparsity_list="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8"

initial_steps=0
step_increment=0
max_iterations=1
is_first_training=1
dataset_type="refined_web"
tasks="gsm8k,humaneval"
num_fewshot=5
seed=2026
limit="${LIMIT:-}"
train_batch_size=1
test_batch_size=2
gradient_accumulation_steps=1
max_seq_length=1024
process_index=1
batch_size="auto:10"
checkpoint_dir=""
results_dir=""
run_training=1
run_evaluation=1
cleanup_weights=1
dry_run=0

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [options]

Required for common runs:
  --cuda DEVICES              CUDA_VISIBLE_DEVICES value, for example 0 or 0,1
  --model-name NAME           Model name, for example sparse_QwenMoE or sparse_OLMoE
  --sparsity-list LIST        Sparsity values, comma or space separated, for example 0.5,0.7,0.8

Useful options:
  --base-model-repo-id PATH   Override the built-in base model path for --model-name
  --tasks TASKS               lm_eval tasks, default: ${tasks}
  --num-fewshot N             Number of few-shot examples, default: ${num_fewshot}
  --seed N                    Random seed, default: ${seed}
  --limit N                   lm_eval --limit value
  --initial-steps N           Initial step count, default: ${initial_steps}
  --step-increment N          Step increment per iteration, default: ${step_increment}
  --max-iterations N          Number of train/eval iterations, default: ${max_iterations}
  --checkpoint-dir PATH       Override checkpoint directory
  --results-dir PATH          Override results directory
  --keep-weights              Keep *.safetensors files under each generated ckpt
  --no-train                  Skip pretrain_sparse_model.py
  --no-eval                   Skip lm_eval
  --dry-run                   Print commands without executing them
  -h, --help                  Show this message

Examples:
  $(basename "$0") --cuda 0 --model-name sparse_QwenMoE --sparsity-list 0.5,0.7,0.8
  $(basename "$0") --cuda 0,1 --model-name sparse_OLMoE --sparsity-list "0.1 0.2 0.3" --tasks gsm8k
EOF
}

default_base_model_repo_id() {
  case "$1" in
    sparse_mixtral_7x8b)
      printf '%s\n' "/home/fit/renju/WORK/lxm/models/Mixtral_8x7B_v0_1"
      ;;
    sparse_llama_7b_hf|sparse_llama_7b_hf2)
      printf '%s\n' "${repo_root}/models/Llama-2-7b-hf"
      ;;
    sparse_QwenMoE)
      printf '%s\n' "${repo_root}/models/models/Qwen1.5-MoE-A2.7B-Chat"
      ;;
    sparse_Deepseek)
      printf '%s\n' "${repo_root}/models/models/DeepSeek_V2_Lite_Chat"
      ;;
    sparse_OLMoE)
      printf '%s\n' "${repo_root}/models/models/OLMoE_1B_7B_0125_Instruct"
      ;;
    *)
      return 1
      ;;
  esac
}

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$dry_run" -eq 0 ]]; then
    "$@"
  fi
}

cleanup_safetensors() {
  local model_dir="$1"

  if [[ "$dry_run" -eq 0 && ! -d "$model_dir" ]]; then
    echo "Skip cleanup: model directory does not exist: ${model_dir}" >&2
    return
  fi

  echo "Cleaning safetensors under: ${model_dir}"
  run_cmd find "$model_dir" -type f -name "*.safetensors" -delete
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda|--cuda-visible-devices)
      cuda_visible_devices="$2"
      shift 2
      ;;
    --model-name)
      model_name="$2"
      shift 2
      ;;
    --sparsity-list)
      sparsity_list="$2"
      shift 2
      ;;
    --base-model-repo-id)
      base_model_repo_id="$2"
      shift 2
      ;;
    --tasks)
      tasks="$2"
      shift 2
      ;;
    --num-fewshot)
      num_fewshot="$2"
      shift 2
      ;;
    --seed)
      seed="$2"
      shift 2
      ;;
    --limit)
      limit="$2"
      shift 2
      ;;
    --initial-steps)
      initial_steps="$2"
      shift 2
      ;;
    --step-increment)
      step_increment="$2"
      shift 2
      ;;
    --max-iterations)
      max_iterations="$2"
      shift 2
      ;;
    --train-batch-size)
      train_batch_size="$2"
      shift 2
      ;;
    --test-batch-size)
      test_batch_size="$2"
      shift 2
      ;;
    --gradient-accumulation-steps)
      gradient_accumulation_steps="$2"
      shift 2
      ;;
    --max-seq-length)
      max_seq_length="$2"
      shift 2
      ;;
    --process-index)
      process_index="$2"
      shift 2
      ;;
    --batch-size)
      batch_size="$2"
      shift 2
      ;;
    --checkpoint-dir)
      checkpoint_dir="$2"
      shift 2
      ;;
    --results-dir)
      results_dir="$2"
      shift 2
      ;;
    --keep-weights)
      cleanup_weights=0
      shift
      ;;
    --no-train)
      run_training=0
      shift
      ;;
    --no-eval)
      run_evaluation=0
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$base_model_repo_id" ]]; then
  if ! base_model_repo_id="$(default_base_model_repo_id "$model_name")"; then
    echo "No default base model path is registered for model_name=${model_name}." >&2
    echo "Please pass --base-model-repo-id explicitly." >&2
    exit 2
  fi
fi

if [[ -z "$checkpoint_dir" ]]; then
  checkpoint_dir="${repo_root}/Sparsity_eval/CATS/${model_name}/ckpt"
fi
if [[ -z "$results_dir" ]]; then
  results_dir="${repo_root}/Sparsity_eval/CATS/${model_name}/result"
fi

read -r -a targeted_sparsity_list <<< "${sparsity_list//,/ }"
if [[ "${#targeted_sparsity_list[@]}" -eq 0 ]]; then
  echo "Empty --sparsity-list." >&2
  exit 2
fi

limit_args=()
if [[ -n "$limit" ]]; then
  limit_args=(--limit "$limit")
fi

run_cmd mkdir -p "$checkpoint_dir" "$results_dir"
export CUDA_VISIBLE_DEVICES="$cuda_visible_devices"

cat <<EOF
========== Zero-shot sparse evaluation ==========
CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}
model_name:           ${model_name}
base_model_repo_id:   ${base_model_repo_id}
sparsity_list:        ${targeted_sparsity_list[*]}
tasks:                ${tasks}
num_fewshot:          ${num_fewshot}
seed:                 ${seed}
limit:                ${limit:-none}
checkpoint_dir:       ${checkpoint_dir}
results_dir:          ${results_dir}
run_training:         ${run_training}
run_evaluation:       ${run_evaluation}
cleanup_weights:      ${cleanup_weights}
dry_run:              ${dry_run}
=================================================
EOF

for targeted_sparsity in "${targeted_sparsity_list[@]}"; do
  sparsity_percentage="$(
    python - "$targeted_sparsity" <<'PY'
import sys
print(f"{float(sys.argv[1]) * 100:.0f}")
PY
  )"
  run_name="${model_name}_${dataset_type}_${sparsity_percentage}p"

  echo "----- sparsity=${targeted_sparsity} (${sparsity_percentage}p) -----"
  for ((i=1; i<=max_iterations; i++)); do
    current_steps=$((initial_steps + i * step_increment))
    model_directory="${checkpoint_dir}/general_finetuning/${run_name}_no_adapter_${current_steps}steps"
    evaluation_output_dir="${results_dir}/evaluations/${model_name}_sparse_${sparsity_percentage}p_${current_steps}steps2"

    if [[ "$run_training" -eq 1 ]]; then
      run_cmd python "${repo_root}/CATS/experiments/pretrain_sparse_model.py" \
        --use_sparse_model --targeted_sparsity "$targeted_sparsity" \
        --set_sparsity_aware_threshold --print_sparsity \
        --seed "$seed" --use_wandb --max_steps "$current_steps" --model_save \
        --train_batch_size "$train_batch_size" --test_batch_size "$test_batch_size" \
        --use_flash_attn --gradient_accumulation_steps "$gradient_accumulation_steps" \
        --ds_config_path ds_config.json --max_seq_length "$max_seq_length" \
        --checkpoint_dir "$checkpoint_dir" --results_dir "$results_dir" \
        --is_first_training "$is_first_training" \
        --gradient_checkpointing \
        --model_name "$model_name" \
        --base_model_repo_id "$base_model_repo_id" \
        --process_index "$process_index" \
        --no-use_lora
    fi

    echo "model directory: ${model_directory}"

    if [[ "$run_evaluation" -eq 1 ]]; then
      run_cmd mkdir -p "$(dirname "$evaluation_output_dir")"
      run_cmd python -m lm_eval \
        --model hf \
        --model_args "pretrained=${model_directory},trust_remote_code=True,parallelize=True,device_map=auto" \
        --tasks "$tasks" \
        --num_fewshot "$num_fewshot" \
        --batch_size "$batch_size" \
        --seed "$seed" \
        "${limit_args[@]}" \
        --confirm_run_unsafe_code \
        --log_samples \
        --output_path "$evaluation_output_dir"
    fi

    if [[ "$cleanup_weights" -eq 1 && "$run_training" -eq 1 ]]; then
      cleanup_safetensors "$model_directory"
    fi
  done
done
