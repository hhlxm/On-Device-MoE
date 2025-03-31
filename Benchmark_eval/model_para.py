import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import tasks, evaluator
from lm_eval.models import huggingface
from lm_eval.utils import setup_logging
import lm_eval
from accelerate import Accelerator

class DeepSeekMMLUEvaluator:
    def __init__(self, model_name='/mnt/petrelfs/liuxinmin/moe/DeepSeek-V2-Lite-Chat'):
        self.model_name = model_name

    def evaluate_mmlu(self,output_file, subtasks=None, num_fewshot=5):
        """
        在 MMLU 数据集上评估模型性能

        参数:
        - subtasks: MMLU 子任务列表 (None 表示所有子任务或字符串表示单个任务)
        - num_fewshot: 少样本学习的样本数
        """
        # 如果未指定子任务，使用所有 MMLU 子任务
        if subtasks is None:
            subtasks = [
                'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
                'clinical_knowledge', 'college_biology', 'college_chemistry',
                'college_computer_science', 'college_mathematics',
                'college_medicine', 'college_physics', 'computer_security',
                'conceptual_physics', 'econometrics', 'electrical_engineering',
                'elementary_mathematics', 'formal_logic', 'global_facts',
                'high_school_biology', 'high_school_chemistry',
                'high_school_computer_science', 'high_school_economics',
                'high_school_geography', 'high_school_government',
                'high_school_mathematics', 'high_school_physics',
                'high_school_psychology', 'high_school_statistics',
                'human_aging', 'human_sexuality', 'international_law',
                'jurisprudence', 'linguistics', 'logical_fallacies',
                'machine_learning', 'management', 'marketing', 'medical_genetics',
                'miscellaneous', 'moral_disputes', 'moral_scenarios',
                'philosophy', 'prehistory', 'professional_accounting',
                'professional_law', 'professional_medicine', 'psychology',
                'public_relations', 'security_studies', 'sociology',
                'us_foreign_policy', 'virology'
            ]
        elif isinstance(subtasks, str):
            subtasks = [subtasks]  # 支持单个任务字符串

        # 配置 MMLU 任务
        task_manager = lm_eval.tasks.TaskManager()

        # 运行评估

        results = lm_eval.simple_evaluate(
                model="hf",
                model_args = f'pretrained={self.model_name},dtype=float16,trust_remote_code=True,parallelize=True',
                tasks=subtasks,
                num_fewshot=num_fewshot,
                task_manager=task_manager,
                batch_size="auto:4",
            )


        self.print_results(results,output_file=output_file)
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
        if output_file :
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
                                f.write(f"{task.upper()} 平均准确率: {acc * 100:.2f}%\n")
                            else:
                                for metric_name, value in metrics.items():
                                    if isinstance(value, (int, float)):
                                        f.write(f"{task.upper()} {metric_name}: {value * 100:.2f}%\n")
                    f.write("\n子任务详细性能:\n")
                    if subtask_found:
                        for task, metrics in sorted(results["results"].items()):
                            if task in aggregate_metrics:
                                continue
                            acc = metrics.get("acc,none", metrics.get("acc", None))
                            if acc is not None:
                                task_name = task.split("_", 1)[1] if "_" in task else task
                                f.write(f"{task_name}: {acc * 100:.2f}%\n")
                            else:
                                for metric_name, value in metrics.items():
                                    if isinstance(value, (int, float)):
                                        task_name = task.split("_", 1)[1] if "_" in task else task
                                        f.write(f"{task_name} ({metric_name}): {value * 100:.2f}%\n")
                    else:
                        f.write("无子任务结果可显示\n")
                print(f"\n结果已保存至: {output_file}")
            else:
                raise ValueError(f"不支持的文件格式: {output_path.suffix}，请使用 .json 或 .txt")
def main():
    # 初始化日志（可选）
    # setup_logging("DEBUG")

    # 创建评估器实例
    evaluator = DeepSeekMMLUEvaluator(
        # model_name='/mnt/hwfile/trustai/share/models/mistralai/Mixtral-8x7B-Instruct-v0.1'
        model_name='/mnt/petrelfs/liuxinmin/moe/models/Phi-3.5-MoE-instruct'
    )

    # 评估 MMLU 子任务j
    results = evaluator.evaluate_mmlu(
        output_file="./results/Phi-3.5-MoE-instruct/result.json",
        subtasks="mmlu",  # 单任务测试
        num_fewshot=5,
        
    )

    # 打印结果
    # evaluator.print_results(results)

if __name__ == "__main__":
    main()