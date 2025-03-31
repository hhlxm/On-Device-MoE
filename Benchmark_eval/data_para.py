import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import tasks, evaluator
from lm_eval.models import huggingface
from lm_eval.utils import setup_logging
import lm_eval
from accelerate import Accelerator

class DeepSeekMMLUEvaluator:
    def __init__(self, model_name='/mnt/petrelfs/liuxinmin/moe/DeepSeek-V2-Lite-Chat'):
        # 初始化 Accelerator
        # self.accelerator = Accelerator(mixed_precision="fp16")
        self.accelerator = Accelerator()

        # 加载模型和分词器
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"  # 自动分配到多个 GPU
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # 配置 lm-eval-harness 模型适配器
        self.lm_model = huggingface.HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
        )
        # 使用 Accelerator 准备模型和分词器
        self.lm_model,self.model,self.tokenizer = self.accelerator.prepare(self.lm_model,self.model,self.tokenizer)

    def evaluate_mmlu(self, subtasks=None, num_fewshot=5):
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
        with self.accelerator.autocast():  # 启用混合精度推理
            results = lm_eval.simple_evaluate(
                model=self.lm_model,
                tasks=subtasks,
                num_fewshot=num_fewshot,
                task_manager=task_manager,
                batch_size="auto:4",
            )


        if not self.accelerator.is_main_process:
            return None  # 非主进程不打印结果
        self.print_results(results)
        return results

    def print_results(self, results):
        # 仅主进程打印结果
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

def main():
    # 初始化日志（可选）
    # setup_logging("DEBUG")

    # 创建评估器实例
    evaluator = DeepSeekMMLUEvaluator(
        model_name='/mnt/hwfile/trustai/share/models/mistralai/Mixtral-8x7B-Instruct-v0.1'
    )

    # 评估 MMLU 子任务
    results = evaluator.evaluate_mmlu(
        subtasks="mmlu_management",  # 单任务测试
        num_fewshot=5,
    )

    # 打印结果
    # evaluator.print_results(results)

if __name__ == "__main__":
    main()