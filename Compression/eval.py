'''
Author: hhlxm 578723542@qq.com
Date: 2025-05-14 01:21:51
LastEditors: hhlxm 578723542@qq.com
LastEditTime: 2025-05-15 14:14:07
FilePath: /lxm/Compression/eval.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json
from pathlib import Path
import argparse
from gptqmodel import GPTQModel
from gptqmodel.utils.eval import EVAL
import torch

def serialize_dict(obj):
    """Handle non-serializable objects for JSON output"""
    if isinstance(obj, dict):
        return {k: serialize_dict(v) for k, v in obj.items()}
    elif isinstance(obj, torch.dtype):
        return str(obj)  # Convert torch.dtype to string
    elif isinstance(obj, (list, tuple)):
        return [serialize_dict(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)  # Convert other types to string

def evaluate_model(model_id,  tasks=None,output_file=None):
    """
    Evaluate a model and save results
    
    Args:
        model_id (str): Model path or identifier
        output_file (str, optional): Output file path for results
        tasks (list, optional): List of evaluation tasks
        trust_remote_code (bool): Whether to trust remote code
    """
    if tasks is None:
        tasks = [EVAL.LM_EVAL.MMLU]
    
    print(f"\nEvaluating model: {model_id}")
    print(f"Tasks: {[task.name if hasattr(task, 'name') else str(task) for task in tasks]}")
    
    # Execute evaluation
    lm_eval_results = GPTQModel.eval(
        model_id, 
        framework=EVAL.LM_EVAL, 
        tasks=tasks,
        batch_size="auto:4",
        num_fewshot=5,
    )
    
    # Save results if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serializable_results = serialize_dict(lm_eval_results)

        if output_path.suffix == ".json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=4, ensure_ascii=False)
            print(f"\nResults saved to: {output_file}")
    
    return lm_eval_results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate a GPTQ model using lm-evaluation-harness")
    
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Path to the model or model identifier"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file path for results (default: model_id.json)"
    )
        
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    tasks = [EVAL.LM_EVAL.MMLU]
    
    print(f"\nStarting evaluation with parameters:")
    print(f"Model ID: {args.model_id}")

    
    results = evaluate_model(
        model_id=args.model_id,
        tasks=tasks,
        output_file=args.output_file,
    )
    
    print("\nEvaluation completed successfully.")

if __name__ == "__main__":
    main()