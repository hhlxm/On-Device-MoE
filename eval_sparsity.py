"""
Evaluate DeepSeek-V2-Lite (sparsity pipeline) with lm-evaluation-harness.

Usage examples:

  # 1) Single GPU (or model parallel across all visible GPUs)
  CUDA_VISIBLE_DEVICES=0 python eval_sparsity.py \
      --model_path /path/to/DeepSeek-V2-Lite \
      --tasks mmlu --num_fewshot 5 --mode none

  # 2) Model parallel only (model sharded across 4 GPUs, single process)
  CUDA_VISIBLE_DEVICES=4,5,6,7 python eval_sparsity.py \
      --model_path /path/to/DeepSeek-V2-Lite \
      --tasks mmlu --sparsity_ratio 0.5 --mode hybrid

  # 3) Data parallel only (model fits on 1 GPU, 4 copies for speed)
  CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 \
      eval_sparsity.py --model_path /path/to/DeepSeek-V2-Lite \
      --tasks mmlu --sparsity_ratio 0.5 --mode hybrid \
      --gpus_per_model 1

  # 4) Model parallel + Data parallel
  #    8 GPUs total, model needs 2 GPUs -> 4 copies for data parallelism
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 4 \
      eval_sparsity.py --model_path /path/to/DeepSeek-V2-Lite \
      --tasks mmlu --sparsity_ratio 0.5 --mode hybrid \
      --gpus_per_model 2

  #    4 GPUs total, model needs 2 GPUs -> 2 copies
  CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 2 \
      eval_sparsity.py --model_path /path/to/DeepSeek-V2-Lite \
      --tasks mmlu --sparsity_ratio 0.5 --mode hybrid \
      --gpus_per_model 2
"""

import argparse
import functools
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lm_eval
from lm_eval.models import huggingface

from models_adapter.deepseek_v2_lite.modeling_deepseek_sparsity_pipeline import (
    DeepseekV2ForCausalLM,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="lm_eval for DeepSeek-V2-Lite with sparsity pipeline"
    )
    # Model
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    p.add_argument("--device_map", type=str, default="auto")

    # Sparsity
    p.add_argument(
        "--sparsity_ratio", type=float, default=0.0,
        help="Neuron sparsity ratio (0.0 = disabled, 0.5 = keep 50%%)",
    )
    p.add_argument(
        "--mode", type=str, default="none",
        choices=["none", "ondemand", "prefetch", "hybrid"],
        help="Sparsity pipeline mode",
    )
    p.add_argument(
        "--prefetch_expert_ratio", type=float, default=0.8,
        help="Fraction of experts to predict in hybrid mode",
    )

    # lm_eval
    p.add_argument(
        "--tasks", type=str, default="mmlu",
        help="Comma-separated task names (e.g. mmlu,hellaswag,winogrande)",
    )
    p.add_argument("--num_fewshot", type=int, default=5)
    p.add_argument("--batch_size", type=str, default="auto:4")
    p.add_argument("--seed", type=int, nargs="*", default=[0, 1234, 1234, 1234],
                   help="seed for python, numpy, torch, fewshot (default: 0 1234 1234 1234)")
    p.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory to save result JSON files",
    )

    # Multi-GPU
    p.add_argument(
        "--gpus_per_model", type=int, default=1,
        help="Number of GPUs per model copy (model parallelism). "
             "Total visible GPUs / gpus_per_model = num data-parallel copies. "
             "Must match --num_processes in accelerate launch.",
    )
    return p.parse_args()


MODE_MAP = {
    "none":     {"prefetch": False, "ondemand": False},
    "ondemand": {"prefetch": False, "ondemand": True},
    "prefetch": {"prefetch": True,  "ondemand": False},
    "hybrid":   {"prefetch": True,  "ondemand": True},
}


def build_output_filename(args):
    """Auto-generate result filename from args.

    Example: DeepSeek-V2-Lite_mmlu_hybrid_sp0.5_ep0.8_5shot.json
    """
    model_name = os.path.basename(args.model_path.rstrip("/\\"))
    tasks_tag = args.tasks.replace(",", "_")
    parts = [model_name, tasks_tag, args.mode]
    if args.mode != "none" and args.sparsity_ratio > 0:
        parts.append(f"sp{args.sparsity_ratio}")
        if args.mode in ("hybrid", "prefetch"):
            parts.append(f"ep{args.prefetch_expert_ratio}")
    parts.append(f"{args.num_fewshot}shot")
    return "_".join(parts) + ".json"


def patch_forward_with_sparsity(model, sparsity_ratio, mode, prefetch_expert_ratio):
    """Monkey-patch model.forward to auto-inject sparsity kwargs."""
    if mode == "none" or sparsity_ratio <= 0.0:
        return  # nothing to patch

    sparsity_kwargs = {
        "neural_sparsity_ratio": sparsity_ratio,
        **MODE_MAP[mode],
        "prefetch_expert_ratio": prefetch_expert_ratio,
    }

    original_forward = model.forward

    @functools.wraps(original_forward)
    def patched_forward(*args, **kwargs):
        for k, v in sparsity_kwargs.items():
            kwargs.setdefault(k, v)
        return original_forward(*args, **kwargs)

    model.forward = patched_forward


def _build_device_map(local_rank, world_size, gpus_per_model, default_device_map):
    """
    Compute device_map for model parallelism + data parallelism.

    With accelerate launch (world_size > 1), each process (local_rank) gets
    a non-overlapping slice of GPUs for model-parallel sharding:
      process 0 → GPU [0, 1, ..., gpus_per_model-1]
      process 1 → GPU [gpus_per_model, ..., 2*gpus_per_model-1]
      ...

    GPU indices are relative to CUDA_VISIBLE_DEVICES.

    Single GPU per model → returns {"": gpu_id} (no sharding).
    Multiple GPUs per model → returns "auto" after restricting visibility.
    Single process → returns default_device_map (usually "auto").
    """
    if world_size <= 1:
        return default_device_map

    if gpus_per_model == 1:
        # Pure data parallel: each process owns exactly one GPU
        return {"": local_rank}

    # Model parallel + data parallel:
    # Restrict each process to its own GPU slice via CUDA_VISIBLE_DEVICES
    start_gpu = local_rank * gpus_per_model
    gpu_ids = list(range(start_gpu, start_gpu + gpus_per_model))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    # After resetting CUDA_VISIBLE_DEVICES, device_map="auto" shards
    # across the visible GPUs (now only this process's slice).
    return "auto"


def main():
    args = parse_args()

    # Detect distributed env from accelerate launch (no Accelerator() needed -
    # lm_eval creates its own internally and they would conflict).
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank == 0

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if is_main:
        print(f"Loading model: {args.model_path}")
        print(f"Sparsity mode: {args.mode}, ratio: {args.sparsity_ratio}")
        print(f"World size: {world_size}, GPUs per model: {args.gpus_per_model}")

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    device_map = _build_device_map(
        local_rank, world_size, args.gpus_per_model, args.device_map
    )

    model = DeepseekV2ForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=dtype_map[args.dtype],
        device_map=device_map,
        trust_remote_code=False
    )
    model.eval()

    if is_main:
        print(f"MoE layers: {model.model.moe_layer_indices}")

    # Inject sparsity kwargs into forward
    patch_forward_with_sparsity(
        model, args.sparsity_ratio, args.mode, args.prefetch_expert_ratio
    )

    # Wrap for lm_eval (HFLM creates its own Accelerator for distributed)
    lm_model = huggingface.HFLM(
        pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size
    )

    # Run evaluation
    tasks = [t.strip() for t in args.tasks.split(",")]
    task_manager = lm_eval.tasks.TaskManager()

    if is_main:
        print(f"Tasks: {tasks}, num_fewshot: {args.num_fewshot}")
        print("-" * 60)

    # Pad seed list to 4 elements if user provided fewer
    seed = args.seed
    if len(seed) == 1:
        seed = seed * 4
    while len(seed) < 4:
        seed.append(seed[-1])

    results = lm_eval.simple_evaluate(
        model=lm_model,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        task_manager=task_manager,
        batch_size=args.batch_size,
        random_seed=seed[0],
        numpy_random_seed=seed[1],
        torch_random_seed=seed[2],
        fewshot_random_seed=seed[3],
        confirm_run_unsafe_code=True
    )

    # Only main process prints and saves
    if not is_main:
        return

    if results and "results" in results:
        for task_name, task_result in results["results"].items():
            print(f"\n[{task_name}]")
            for metric, value in task_result.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")

    # Save only summary results, drop per-sample records
    save_data = {k: v for k, v in results.items() if k != "samples"}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / build_output_filename(args)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
