"""
Test script for DeepSeek-V2-Lite with neuron sparsity pipeline.

Usage:
    python test.py --model_path /path/to/DeepSeek-V2-Lite \
                   --sparsity_ratio 0.5 \
                   --mode hybrid
"""

import argparse
import sys
import os
import time

import torch
from transformers import AutoTokenizer, AutoConfig

# Add project root so we can import the custom modeling file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models_adapter.deepseek_v2_lite.modeling_deepseek_sparsity_pipeline import (
    DeepseekV2ForCausalLM,
)


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeek-V2-Lite sparsity test")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Local path to DeepSeek-V2-Lite model weights",
    )
    parser.add_argument(
        "--prompt", type=str,
        default="The meaning of life is",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=50,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--sparsity_ratio", type=float, default=0.5,
        help="Neuron sparsity ratio (0.5 = keep top 50%%)",
    )
    parser.add_argument(
        "--mode", type=str, default="hybrid",
        choices=["none", "ondemand", "prefetch", "hybrid"],
        help="Sparsity pipeline mode",
    )
    parser.add_argument(
        "--prefetch_expert_ratio", type=float, default=0.8,
        help="Fraction of experts to predict in hybrid mode",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on (cuda / cpu)",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    return parser.parse_args()


def get_sparsity_kwargs(args):
    """Convert CLI mode into forward() keyword arguments."""
    if args.mode == "none" or args.sparsity_ratio < 0:
        return {}
    mode_map = {
        "ondemand": {"prefetch": False, "ondemand": True},
        "prefetch": {"prefetch": True, "ondemand": False},
        "hybrid":   {"prefetch": True, "ondemand": True},
    }
    return {
        "neural_sparsity_ratio": args.sparsity_ratio,
        **mode_map[args.mode],
        "prefetch_expert_ratio": args.prefetch_expert_ratio,
    }


@torch.no_grad()
def generate(model, tokenizer, input_ids, max_new_tokens, sparsity_kwargs, device):
    """Simple greedy generation loop with sparsity support."""
    past_key_values = None
    generated_ids = input_ids.clone()

    for step in range(max_new_tokens):
        if past_key_values is None:
            # Prefill: full sequence, no sparsity
            outputs = model(
                input_ids=generated_ids,
                use_cache=True,
            )
        else:
            # Decode: single token with sparsity
            outputs = model(
                input_ids=generated_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
                **sparsity_kwargs,
            )

        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return generated_ids


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(42)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading model from: {args.model_path}")
    print(f"Device: {args.device}, dtype: {args.dtype}")

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = DeepseekV2ForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    print(f"Model loaded. MoE layers: {model.model.moe_layer_indices}")
    print(f"Mode: {args.mode}, sparsity_ratio: {args.sparsity_ratio}")
    print(f"Prompt: {args.prompt!r}")
    print("-" * 60)

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(args.device)
    sparsity_kwargs = get_sparsity_kwargs(args)

    # Warmup
    _ = generate(model, tokenizer, input_ids, 2, sparsity_kwargs, args.device)
    if args.device == "cuda":
        torch.cuda.synchronize()

    # Timed generation
    start = time.perf_counter()
    output_ids = generate(
        model, tokenizer, input_ids, args.max_new_tokens,
        sparsity_kwargs, args.device,
    )
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    n_new = output_ids.shape[1] - input_ids.shape[1]

    print(f"Output: {output_text}")
    print("-" * 60)
    print(f"Generated {n_new} tokens in {elapsed:.3f}s "
          f"({n_new / elapsed:.1f} tok/s)")

    # Baseline comparison (no sparsity)
    if args.mode != "none" and args.sparsity_ratio > 0:
        print("\n--- Baseline (no sparsity) ---")
        start = time.perf_counter()
        baseline_ids = generate(
            model, tokenizer, input_ids, args.max_new_tokens, {}, args.device,
        )
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed_base = time.perf_counter() - start

        baseline_text = tokenizer.decode(baseline_ids[0], skip_special_tokens=True)
        n_base = baseline_ids.shape[1] - input_ids.shape[1]
        print(f"Output: {baseline_text}")
        print(f"Generated {n_base} tokens in {elapsed_base:.3f}s "
              f"({n_base / elapsed_base:.1f} tok/s)")

        # Check divergence
        min_len = min(output_ids.shape[1], baseline_ids.shape[1])
        match = (output_ids[0, :min_len] == baseline_ids[0, :min_len]).sum().item()
        print(f"Token match: {match}/{min_len} "
              f"(first divergence at position {match} if < {min_len})")


if __name__ == "__main__":
    main()
