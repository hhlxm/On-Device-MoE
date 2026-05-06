"""
Evaluate MoE models (sparsity pipeline) with lm-evaluation-harness.
Supports: DeepSeek-V2-Lite, OLMoE-1B-7B-0125-Instruct, Qwen1.5-MoE-A2.7B

Usage examples:

  # DeepSeek-V2-Lite
  python eval_sparsity.py --model_type deepseek \
      --model_path /path/to/DeepSeek-V2-Lite \
      --tasks mmlu --sparsity_ratio 0.5 --mode hybrid \
      --limit 100

  # OLMoE-1B-7B
  python eval_sparsity.py --model_type olmoe \
      --model_path /path/to/OLMoE-1B-7B-0125-Instruct \
      --tasks mmlu --sparsity_ratio 0.5 --mode hybrid

  # Qwen1.5-MoE-A2.7B
  python eval_sparsity.py --model_type qwen \
      --model_path /path/to/Qwen1.5-MoE-A2.7B-Chat \
      --tasks mmlu --sparsity_ratio 0.5 --mode hybrid

  # Data parallel (4 copies)
  CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 \
      eval_sparsity.py --model_type deepseek \
      --model_path /path/to/DeepSeek-V2-Lite \
      --tasks mmlu --sparsity_ratio 0.5 --mode hybrid \
      --gpus_per_model 1

  # Model parallel + Data parallel (8 GPUs, 2 per model -> 4 copies)
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 4 \
      eval_sparsity.py --model_type deepseek \
      --model_path /path/to/DeepSeek-V2-Lite \
      --tasks mmlu --sparsity_ratio 0.5 --mode hybrid \
      --gpus_per_model 2
"""

import argparse
import functools
import json
import os
import sys
from pathlib import Path

from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lm_eval
from lm_eval.models import huggingface

from models_adapter.deepseek_v2_lite.configuration_deepseek import DeepseekV2Config
from models_adapter.deepseek_v2_lite.modeling_deepseek_sparsity_pipeline import (
    DeepseekV2ForCausalLM,
)
from models_adapter.olmoe_1b_7b_0125_instruct.configuration_olmoe import OlmoeConfig
from models_adapter.olmoe_1b_7b_0125_instruct.modeling_olmoe_sparsity_pipeline import (
    OlmoeForCausalLM,
)

from models_adapter.qwen_1_5_moe_a2_7b.configuration_qwen2_moe import Qwen2MoeConfig
from models_adapter.qwen_1_5_moe_a2_7b.modeling_qwen2_moe_sparsity_pipeline import (
    Qwen2MoeForCausalLM,
)


def parse_limit(value):
    """Parse lm-eval limit: integer sample count or 0-1 dataset fraction."""
    if value is None:
        return None

    try:
        if "." not in value:
            limit = int(value)
            if limit <= 0:
                raise ValueError
            return limit

        limit = float(value)
        if not 0 < limit <= 1:
            raise ValueError
        return limit
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--limit must be a positive integer sample count, or a float in (0, 1]."
        ) from exc


def parse_args():
    p = argparse.ArgumentParser(
        description="lm_eval for MoE models with sparsity pipeline"
    )
    # Model
    p.add_argument(
        "--model_type", type=str, default="deepseek",
        choices=["deepseek", "olmoe", "qwen"],
        help="Model type: deepseek, olmoe, or qwen (Qwen1.5-MoE-A2.7B)",
    )
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )

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
    p.add_argument(
        "--limit",
        type=parse_limit,
        default=None,
        help="Limit evaluation data per task. Use an integer for sample count "
             "(e.g. 100), or a float in (0, 1] for dataset fraction (e.g. 0.1).",
    )
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
    if args.limit is not None:
        parts.append(f"limit{args.limit}")
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


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank == 0

    if is_main:
        print(f"Loading model ({args.model_type}): {args.model_path}")
        print(f"Sparsity mode: {args.mode}, ratio: {args.sparsity_ratio}")
        print(f"World size: {world_size}, GPUs per model: {args.gpus_per_model}")

    # ---- Register custom model classes so AutoModelForCausalLM can find them ----
    AutoConfig.register("deepseek_v2", DeepseekV2Config, exist_ok=True)
    AutoModelForCausalLM.register(DeepseekV2Config, DeepseekV2ForCausalLM, exist_ok=True)
    AutoConfig.register("olmoe", OlmoeConfig, exist_ok=True)
    AutoModelForCausalLM.register(OlmoeConfig, OlmoeForCausalLM, exist_ok=True)
    AutoConfig.register("qwen2_moe", Qwen2MoeConfig, exist_ok=True)
    AutoModelForCausalLM.register(Qwen2MoeConfig, Qwen2MoeForCausalLM, exist_ok=True)

    # For model parallel + data parallel: restrict each process to its GPU slice
    # BEFORE HFLM creates its Accelerator, so device_map="auto" shards correctly.
    #
    # Bug fix: read the *current* CUDA_VISIBLE_DEVICES (physical IDs set by the
    # user, e.g. "4,5,6,7") and slice it, rather than generating logical indices
    # with range() which would incorrectly point to physical GPUs 0,1,... instead
    # of the user-specified ones.
    if world_size > 1 and args.gpus_per_model > 1:
        visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible_env:
            physical_ids = [x.strip() for x in visible_env.split(",") if x.strip()]
        else:
            import torch
            physical_ids = [str(i) for i in range(torch.cuda.device_count())]

        total_visible = len(physical_ids)
        expected_procs = total_visible // args.gpus_per_model
        if total_visible % args.gpus_per_model != 0:
            raise RuntimeError(
                f"Total visible GPUs ({total_visible}) is not divisible by "
                f"gpus_per_model ({args.gpus_per_model}). "
                f"Visible GPUs: {physical_ids}"
            )
        if world_size != expected_procs:
            raise RuntimeError(
                f"--num_processes should be {expected_procs} "
                f"({total_visible} GPUs / {args.gpus_per_model} per model), "
                f"but got world_size={world_size}"
            )

        start = local_rank * args.gpus_per_model
        my_ids = physical_ids[start:start + args.gpus_per_model]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(my_ids)
        if is_main:
            print(f"GPU slicing: {total_visible} GPUs -> {world_size} copies, "
                  f"rank {local_rank} uses physical GPUs {my_ids}")

    # ---- Create HFLM with string path (enables full distributed support) ----
    hflm_kwargs = dict(
        pretrained=args.model_path,
        dtype=args.dtype,
        batch_size=args.batch_size,
        trust_remote_code=False,
    )
    if args.gpus_per_model > 1:
        hflm_kwargs["device_map"] = "auto"

    lm_model = huggingface.HFLM(**hflm_kwargs)

    if is_main:
        print(f"HFLM rank={lm_model.rank}, world_size={lm_model.world_size}")
        if hasattr(lm_model._model, "model") and hasattr(lm_model._model.model, "moe_layer_indices"):
            print(f"MoE layers: {lm_model._model.model.moe_layer_indices}")

    # Inject sparsity kwargs into the underlying model's forward
    patch_forward_with_sparsity(
        lm_model._model, args.sparsity_ratio, args.mode, args.prefetch_expert_ratio
    )

    # ---- Run evaluation ----
    tasks = [t.strip() for t in args.tasks.split(",")]
    task_manager = lm_eval.tasks.TaskManager()

    if is_main:
        print(f"Tasks: {tasks}, num_fewshot: {args.num_fewshot}, limit: {args.limit}")
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
        limit=args.limit,
        confirm_run_unsafe_code=True
    )

    # Only main process prints and saves (use lm_model.rank for correctness)
    if lm_model.rank != 0:
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
