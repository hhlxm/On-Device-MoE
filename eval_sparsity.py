"""
Evaluate DeepSeek-V2-Lite (sparsity pipeline) with lm-evaluation-harness.

Usage examples:

  # Baseline (no sparsity)
  python eval_sparsity.py --model_path /path/to/DeepSeek-V2-Lite \
      --tasks mmlu --num_fewshot 5 --mode none

  # On-demand sparsity, 50% neurons pruned
  python eval_sparsity.py --model_path /path/to/DeepSeek-V2-Lite \
      --tasks mmlu,hellaswag --sparsity_ratio 0.5 --mode ondemand

  # Hybrid mode (prefetch + ondemand fallback)
  python eval_sparsity.py --model_path /path/to/DeepSeek-V2-Lite \
      --tasks mmlu --sparsity_ratio 0.5 --mode hybrid \
      --prefetch_expert_ratio 0.8

  # Pure prefetch mode
  python eval_sparsity.py --model_path /path/to/DeepSeek-V2-Lite \
      --tasks mmlu --sparsity_ratio 0.5 --mode prefetch
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
    p.add_argument("--output_file", type=str, default=None)
    return p.parse_args()


MODE_MAP = {
    "none":     {"prefetch": False, "ondemand": False},
    "ondemand": {"prefetch": False, "ondemand": True},
    "prefetch": {"prefetch": True,  "ondemand": False},
    "hybrid":   {"prefetch": True,  "ondemand": True},
}


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

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    print(f"Loading model: {args.model_path}")
    print(f"Sparsity mode: {args.mode}, ratio: {args.sparsity_ratio}")

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = DeepseekV2ForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    print(f"MoE layers: {model.model.moe_layer_indices}")

    # Inject sparsity kwargs into forward
    patch_forward_with_sparsity(
        model, args.sparsity_ratio, args.mode, args.prefetch_expert_ratio
    )

    # Wrap for lm_eval
    lm_model = huggingface.HFLM(
        pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size
    )

    # Run evaluation
    tasks = [t.strip() for t in args.tasks.split(",")]
    task_manager = lm_eval.tasks.TaskManager()

    print(f"Tasks: {tasks}, num_fewshot: {args.num_fewshot}")
    print("-" * 60)

    results = lm_eval.simple_evaluate(
        model=lm_model,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        task_manager=task_manager,
        batch_size=args.batch_size,
    )

    # Print results
    if results and "results" in results:
        for task_name, task_result in results["results"].items():
            print(f"\n[{task_name}]")
            for metric, value in task_result.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")

    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
