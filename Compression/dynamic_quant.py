import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np


# 添加当前目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from accelerate import Accelerator
from models.DeepSeek_V2_Lite.modeling_deepseek_quant import DeepseekV2ForCausalLM
from models.Mixtral_8x7B_v0_1.modeling_mixtral_quant import MixtralForCausalLM
from models.Phi_3_5_MoE_instruct.modeling_phimoe_quant import PhiMoEForCausalLM
from models.Qwen1_5_MoE_A2_7B.modeling_qwen2_moe_quant import Qwen2MoeForCausalLM
from transformers import AutoTokenizer

import lm_eval

# from models.DeepSeek_V2_Lite.modeling_deepseek_prefetch import DeepseekV2ForCausalLM
# from ...models.DeepSeek_V2_Lite.modeling_deepseek import DeepseekV2ForCausalLM
from lm_eval.models import huggingface


class Evaluator:
    def __init__(self, args, model_path, checkpoint_path, batch_size="auto:4"):
        """
        Initialize the evaluator with configurable model and checkpoint paths.

        Parameters:
        - model_path: Path to the pretrained model
        - checkpoint_path: Path to the model checkpoint
        - batch_size: Batch size for evaluation (default: "auto:4")
        """
        self.accelerator = Accelerator()

        # Load model and tokenizer

        # if checkpoint_path is not None:
        #     from models.DeepSeek_V2_Lite.modeling_deepseek_pregate import DeepseekV2ForCausalLM
        #     self.model = DeepseekV2ForCausalLM.from_pretrained(
        #     model_path,
        #     trust_remote_code=True,
        #     torch_dtype=torch.float16,
        #     device_map="auto"
        #     )

        # else:

        if "deepseek" in model_path.lower():
            self.model = DeepseekV2ForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            # checkpoint = torch.load(checkpoint_path, map_location="cuda")
            # self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif "phi" in model_path.lower():
            self.model = PhiMoEForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        elif "qwen" in model_path.lower():
            self.model = Qwen2MoeForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        elif "mixtral" in model_path.lower():
            self.model = MixtralForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        if self.model.model.quant_wbit is None:
            raise ValueError("quant_wbit is None, please check the model")
        if self.model.model.quant_num is None:
            raise ValueError("quant_num is None, please check the model")
        self.model.model.quant_wbit = args.quant_wbit
        self.model.model.quant_num = args.quant_num
        print("quant_wbit:", self.model.model.quant_wbit)
        print("quant_num:", self.model.model.quant_num)
        # self.model.model.pre_dis=args.pre_dis
        # self.model.model.pre_ahead=args.pre_ahead
        # print("pre_dis:",self.model.model.pre_dis)
        # print("pre_ahead:",self.model.model.pre_ahead)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # Configure lm-eval-harness model adapter
        self.lm_model = huggingface.HFLM(
            pretrained=self.model, tokenizer=self.tokenizer, batch_size="auto:4"
        )

    def evaluate(self, subtasks=None, num_fewshot=5, output_file=None):
        """
        在 MMLU 数据集上评估模型性能

        参数:
        - subtasks: MMLU 子任务列表 (None 表示所有子任务或字符串表示单个任务)
        - num_fewshot: 少样本学习的样本数
        """
        # 如果未指定子任务，使用所有 MMLU 子任务

        # 配置 MMLU 任务
        task_manager = lm_eval.tasks.TaskManager()

        # 运行评估
        with self.accelerator.autocast():  # 启用混合精度推理
            results = lm_eval.simple_evaluate(
                model=self.lm_model,
                tasks=subtasks,
                num_fewshot=num_fewshot,
                task_manager=task_manager,
                batch_size="auto:4",
                confirm_run_unsafe_code=True,
            )

        if not self.accelerator.is_main_process:
            return None  # 非主进程不打印结果
        self.print_results(results, output_file)
        return results

    def print_results(self, results, output_file=None):
        """
        格式化打印评估结果，并可选保存到文件。

        参数:
        - results: 评估结果字典
        - output_file: 保存结果的文件路径（可选，支持 .json、.txt 等格式）
        """
        # 仅主进程执行打印和保存
        if results is None:
            return

        print("\n===== 评估结果 =====")

        if not results or "results" not in results:
            print("无有效结果可打印")
            return

        # 打印总体性能
        aggregate_metrics = {}
        for task, metrics in results["results"].items():
            if task == "mmlu" or not task.startswith(tuple(aggregate_metrics.keys())):
                aggregate_metrics[task] = metrics

        if aggregate_metrics:
            print("\n总体性能:")
            for task, metrics in aggregate_metrics.items():
                acc = metrics.get("acc,none", metrics.get("acc", None))
                if acc is not None:
                    print(f"{task.upper()} 平均准确率: {acc * 100:.2f}%")
                else:
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"{task.upper()} {metric_name}: {value * 100:.2f}%")

        # 打印子任务详细结果
        print("\n子任务详细性能:")
        subtask_found = False
        for task, metrics in sorted(results["results"].items()):
            if task in aggregate_metrics:
                continue
            subtask_found = True
            acc = metrics.get("acc,none", metrics.get("acc", None))
            if acc is not None:
                task_name = task.split("_", 1)[1] if "_" in task else task
                print(f"{task_name}: {acc * 100:.2f}%")
            else:
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        task_name = task.split("_", 1)[1] if "_" in task else task
                        print(f"{task_name} ({metric_name}): {value * 100:.2f}%")

        if not subtask_found:
            print("无子任务结果可显示")

        # 保存结果到文件
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 定义一个函数来处理不可序列化的对象
            def serialize_dict(obj):
                if isinstance(obj, dict):
                    return {k: serialize_dict(v) for k, v in obj.items()}
                elif isinstance(obj, torch.dtype):
                    return str(obj)  # 将 torch.dtype 转换为字符串
                elif isinstance(obj, (list, tuple)):
                    return [serialize_dict(item) for item in obj]
                elif isinstance(obj, (int, float, str, bool)) or obj is None:
                    return obj
                else:
                    return str(obj)  # 其他类型默认转为字符串

            # 处理 results 以确保可序列化
            serializable_results = serialize_dict(results)

            # 根据文件扩展名保存
            if output_path.suffix == ".json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(serializable_results, f, indent=4, ensure_ascii=False)
                print(f"\n结果已保存至: {output_file}")
            elif output_path.suffix == ".txt":
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("===== 评估结果 =====\n\n")
                    if aggregate_metrics:
                        f.write("总体性能:\n")
                        for task, metrics in aggregate_metrics.items():
                            acc = metrics.get("acc,none", metrics.get("acc", None))
                            if acc is not None:
                                f.write(
                                    f"{task.upper()} 平均准确率: {acc * 100:.2f}%\n"
                                )
                            else:
                                for metric_name, value in metrics.items():
                                    if isinstance(value, (int, float)):
                                        f.write(
                                            f"{task.upper()} {metric_name}: {value * 100:.2f}%\n"
                                        )
                    f.write("\n子任务详细性能:\n")
                    if subtask_found:
                        for task, metrics in sorted(results["results"].items()):
                            if task in aggregate_metrics:
                                continue
                            acc = metrics.get("acc,none", metrics.get("acc", None))
                            if acc is not None:
                                task_name = (
                                    task.split("_", 1)[1] if "_" in task else task
                                )
                                f.write(f"{task_name}: {acc * 100:.2f}%\n")
                            else:
                                for metric_name, value in metrics.items():
                                    if isinstance(value, (int, float)):
                                        task_name = (
                                            task.split("_", 1)[1]
                                            if "_" in task
                                            else task
                                        )
                                        f.write(
                                            f"{task_name} ({metric_name}): {value * 100:.2f}%\n"
                                        )
                    else:
                        f.write("无子任务结果可显示\n")
                print(f"\n结果已保存至: {output_file}")
            else:
                raise ValueError(
                    f"不支持的文件格式: {output_path.suffix}，请使用 .json 或 .txt"
                )


def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate DeepSeek model on MMLU dataset"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the pretrained model"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--subtasks",
        type=str,
        default="gsm8k",
        help="MMLU subtask to evaluate (default: gsm8k)",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=5,
        help="Number of few-shot examples (default: 5)",
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default="auto:4",
        help="Batch size for evaluation (default: auto:4)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="File to save evaluation results (default: None)",
    )
    parser.add_argument(
        "--pre_dis", type=int, default="1", help="prediction distance (default: 1)"
    )
    parser.add_argument(
        "--pre_ahead", type=int, default="1", help="prediction ahead (default: 1)"
    )

    parser.add_argument(
        "--quant_wbit", type=int, default=0, help="prediction ahead (default: 1)"
    )
    parser.add_argument(
        "--quant_num", type=int, default=0, help="prediction ahead (default: 1)"
    )

    args = parser.parse_args()

    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create evaluator instance
    evaluator = Evaluator(
        args,
        model_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
    )

    # Evaluate MMLU subtask
    evaluator.evaluate(
        subtasks=args.subtasks,
        num_fewshot=args.num_fewshot,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
