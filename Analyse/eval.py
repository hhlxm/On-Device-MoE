import json
import logging
import os
import shutil
import numpy as np
import yaml
import random
import torch
from tqdm import tqdm
import copy
import sys

# 假设这些函数定义在 utli.py 中
from load_models import load_model_and_tokenizer
from my_dataload import load_dataset_sample


def setup_logging(output_dir, model_short_name, dataset_name):
    """设置日志，保存到文件并输出到控制台"""
    log_file = os.path.join(output_dir, model_short_name, dataset_name, "log.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # 保存到文件
            logging.StreamHandler(sys.stdout),  # 输出到控制台
        ],
    )


def load_config(config_path):
    """读取 YAML 配置文件"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config




def cache_process(model, tokenizer, texts, max_new_tokens, device="cuda", hot=True):
    """处理文本并收集缓存统计数据（LRU、FIFO、LFU等），按层存储总命中和请求次数"""

    # 初始化按层存储的总体统计
    overall_stats = {
        "lru": {
            "total_hits": {},  # {layer_idx: total_hits}
            "total_requests": {},  # {layer_idx: total_requests}
            "hit_rates": {},  # {layer_idx: hit_rate}
            "overall_hit_rate": 0.0,
        },
        "fifo": {
            "total_hits": {},
            "total_requests": {},
            "hit_rates": {},
            "overall_hit_rate": 0.0,
        },
        "lfu": {
            "total_hits": {},
            "total_requests": {},
            "hit_rates": {},
            "overall_hit_rate": 0.0,
        },
        "lru_real": {
            "total_hits": {},
            "total_requests": {},
            "hit_rates": {},
            "overall_hit_rate": 0.0,
        },
        "fifo_real": {
            "total_hits": {},
            "total_requests": {},
            "hit_rates": {},
            "overall_hit_rate": 0.0,
        },
        "lfu_real": {
            "total_hits": {},
            "total_requests": {},
            "hit_rates": {},
            "overall_hit_rate": 0.0,
        },
    }

    # 重置模型的所有缓存模拟器
    model.model.reset_all_cache_simulators()

    # 设置分词器的填充标记
    tokenizer.pad_token = tokenizer.eos_token

    for idx, text in enumerate(tqdm(texts, desc="Processing texts")):
        if hot:
            # 重置模型，热启动
            model.model.random_fill_all_cache()
        else:
            # 重置模型，冷启动
            model.model.reset_all_cache_simulators()

        # 编码输入文本
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
            add_special_tokens=True,
        )
        input_ids = inputs["input_ids"].to(device)

        # 打印输入形状
        print(f"idx: {idx}, input shape: {input_ids.shape}")

        # 生成输出
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.85,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.05,
            )
            print(f"output shape: {outputs.shape}")

        # 检查输出是否过短
        if outputs.shape[1] - input_ids.shape[1] <= 1:
            print(f"Skipped - Text: {text}")
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Output: {output_text}")
            continue

        # 收集缓存统计数据并累加到 overall_stats
        cache_types = [
            ("lru", model.model.get_all_lru_cache_stats),
            ("fifo", model.model.get_all_fifo_cache_stats),
            ("lfu", model.model.get_all_lfu_cache_stats),
            ("lru_real", model.model.get_all_lru_cache_real_stats),
            ("fifo_real", model.model.get_all_fifo_cache_real_stats),
            ("lfu_real", model.model.get_all_lfu_cache_real_stats),
        ]

        for cache_type, get_stats_func in cache_types:
            try:
                stats = get_stats_func()
                # 累加每层的统计
                for layer_idx, layer_stats in stats.items():
                    if layer_idx not in overall_stats[cache_type]["total_hits"]:
                        overall_stats[cache_type]["total_hits"][layer_idx] = 0
                        overall_stats[cache_type]["total_requests"][layer_idx] = 0
                    overall_stats[cache_type]["total_hits"][layer_idx] += layer_stats[
                        "hits"
                    ]
                    overall_stats[cache_type]["total_requests"][layer_idx] += (
                        layer_stats["requests"]
                    )
            except AttributeError:
                print(f"Warning: `{get_stats_func.__name__}` not found for run {idx}.")

    # 计算每层和整体的命中率
    for cache_type in overall_stats:
        total_hits_sum = sum(overall_stats[cache_type]["total_hits"].values())
        total_requests_sum = sum(overall_stats[cache_type]["total_requests"].values())
        overall_stats[cache_type]["overall_hit_rate"] = (
            total_hits_sum / total_requests_sum if total_requests_sum > 0 else 0.0
        )
        # 计算每层的命中率
        for layer_idx in overall_stats[cache_type]["total_hits"]:
            hits = overall_stats[cache_type]["total_hits"][layer_idx]
            requests = overall_stats[cache_type]["total_requests"][layer_idx]
            if layer_idx not in overall_stats[cache_type]["hit_rates"]:
                overall_stats[cache_type]["hit_rates"][layer_idx] = 0.0
            overall_stats[cache_type]["hit_rates"][layer_idx] = (
                hits / requests if requests > 0 else 0.0
            )

    # 打印每层和整体的命中率
    print("\n--- Overall Cache Statistics ---")
    for cache_type in overall_stats:
        print(f"\n{cache_type.upper()} Cache:")
        for layer_idx in sorted(overall_stats[cache_type]["total_hits"].keys()):
            print(
                f"  Layer {layer_idx}: Total Hits={overall_stats[cache_type]['total_hits'][layer_idx]}, "
                f"Total Requests={overall_stats[cache_type]['total_requests'][layer_idx]}, "
                f"Hit Rate={overall_stats[cache_type]['hit_rates'][layer_idx]:.4f}"
            )
        print(
            f"  Overall: Total Hits={sum(overall_stats[cache_type]['total_hits'].values())}, "
            f"Total Requests={sum(overall_stats[cache_type]['total_requests'].values())}, "
            f"Hit Rate={overall_stats[cache_type]['overall_hit_rate']:.4f}"
        )

    # 重置模型状态
    model.model.reset_all_cache_simulators()

    # 返回所有统计结果
    return overall_stats


def process_texts(model, tokenizer, texts, max_new_tokens, device="cuda"):
    """处理文本并收集频率数据"""
    frequencies1 = {}
    frequencies2 = {}
    frequencies3 = {}
    frequencies4 = {}

    model.reset_all_expert_counts()
    model.reset_all_layer_top_avg()
    model.reset_all_expert_hit_rate()
    model.reset_all_token_frequency()

    for idx, text in enumerate(tqdm(texts, desc="Processing texts")):
        model.reset_all_expert_hit_rate()
        model.reset_all_token_frequency()

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)

        print(f"idx: {idx}, input shape: {input_ids.shape}")

        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=max_new_tokens)
            print(f"output shape: {outputs.shape}")

        if outputs.shape[1] - input_ids.shape[1] <= 1:
            print(f"Skipped - Text: {text}")
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Output: {output_text}")
            continue

        frequencies3[idx] = copy.deepcopy(model.get_all_expert_hit_rate())
        frequencies4[idx] = copy.deepcopy(model.get_all_token_frequency())

    frequencies1 = copy.deepcopy(model.get_all_expert_frequencies())
    frequencies2 = copy.deepcopy(model.get_all_layer_top_avg())
    model.reset_all_expert_counts()
    model.reset_all_layer_top_avg()
    model.reset_all_expert_hit_rate()
    model.reset_all_token_frequency()
    return frequencies1, frequencies2, frequencies3, frequencies4


def build_activation_matrix(
    model, tokenizer, texts, num_layers, num_experts, max_new_tokens, device
):
    model.reset_all_expert_counts()
    model.reset_all_expert_hit_rate()
    model.reset_all_token_frequency()
    """构建 activation_matrix 并处理文本"""
    activation_matrix = np.zeros((len(texts), num_layers, num_experts))

    for text_idx, input_text in enumerate(
        tqdm(texts, desc="Building activation matrix")
    ):
        print(f"text_idx: {text_idx}")
        if not input_text.strip():  # 跳过空文本
            continue

        model.reset_all_expert_counts()

        inputs = tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"].to(device)
        print(f"input shape: {input_ids.shape}")

        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=max_new_tokens)
            print(f"output shape: {outputs.shape}")

        frequencies1 = copy.deepcopy(model.get_all_expert_frequencies())

        # 填充矩阵
        experts_i = frequencies1[1]["routed"].keys()  # 假设所有层的专家索引一致
        layer_counter = 0
        for layer_idx in sorted(frequencies1.keys()):
            for expert_idx in experts_i:
                activation_matrix[text_idx, layer_counter, expert_idx] = frequencies1[
                    layer_idx
                ]["routed"][expert_idx]
            layer_counter += 1
    model.reset_all_expert_counts()
    model.reset_all_expert_hit_rate()
    model.reset_all_token_frequency()
    return activation_matrix


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_random_seed(2025)
    config_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/home/fit/renju/WORK/lxm/Analyse/config/config.yaml"
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file {config_path} not found. Please provide a valid YAML config file."
        )
    config = load_config(config_path)
    # 读取 YAML 配置

    # 解析配置
    model_config = config.get("model", {})
    dataset_config = config.get("dataset", {})
    generation_config = config.get("generation", {})
    output_config = config.get("output", {})
    analysis_config = config.get("analysis", {})
    matrix_config = config.get("matrix", {})

    # 创建输出目录
    model_short_name = os.path.basename(model_config["name"]).lower()
    dataset_name = dataset_config["name"].lower()
    # request_dir = os.path.join(output_config["dir"], model_short_name,dataset_name, "Request_level")
    # cache_dir = os.path.join(output_config["dir"], model_short_name,dataset_name, "Cache_hit")
    # token_dir = os.path.join(output_config["dir"], model_short_name,dataset_name, "Token_level")
    # matrix_dir = os.path.join(output_config["dir"], model_short_name,dataset_name, "Sequence_level")
    cache_dir = os.path.join(
        output_config["dir"],
        model_short_name,
        dataset_name,
        f"Cache_hit_ratio_{model_config.get('cache_ratio', 0.5)}",
    )

    # 确保所有目录存在
    for dir_path in [cache_dir]:
        os.makedirs(dir_path, exist_ok=True)

    setup_logging(output_config["dir"], model_short_name, dataset_name)
    logging.info(f"Running with config: {yaml.dump(config)}")

    # 复制配置文件到输出目录
    config_output_path = os.path.join(
        output_config["dir"], model_short_name, dataset_name, "config.yaml"
    )
    shutil.copy(config_path, config_output_path)

    # 加载模型和分词器
    model, tokenizer, _ = load_model_and_tokenizer(
        model_config, model_config.get("device", "cuda")
    )

    # 加载数据集
    texts = load_dataset_sample(dataset_name, dataset_config["sample_size"])

    # # 处理文本并收集频率数据
    # try:
    #     frequencies1,frequencies2, frequencies3, frequencies4 = process_texts(
    #         model, tokenizer, texts, generation_config["max_new_tokens"], model_config["device"]
    #     )
    # except Exception as e:
    #     logging.error(f"Failed to process texts: {str(e)}", exc_info=True)
    #     raise

    # # 绘制并保存图表
    # draw(
    #     frequencies1,
    #     title="Expert Activation Heatmap",
    #     output_path=os.path.join(request_dir, f"{dataset_name}.svg")
    # )
    # logging.info(f"Saved Expert Activation Heatmap to {request_dir}")
    # del frequencies1
    # torch.cuda.empty_cache()

    # plot_layer_sums(
    #     frequencies2,
    #     title="Layer Top_sum Average",
    #     output_path=os.path.join(matrix_dir, f"{dataset_name}.svg")
    # )
    # logging.info(f"Saved Layer Top_sum Average to {matrix_dir}")
    # del frequencies2
    # torch.cuda.empty_cache()

    # draw_batch(
    #     frequencies3,
    #     title="Cache Hit Rate",
    #     output_path=os.path.join(cache_dir, f"{dataset_name}.svg")
    # )
    # logging.info(f"Saved Cache Hit Rate to {cache_dir}")
    # del frequencies3
    # torch.cuda.empty_cache()

    # plot_cosine_similarity_avg_cross_batch(
    #     frequencies4,
    #     layers=range(analysis_config["layer_start"], analysis_config["layer_end"]),
    #     token_start=analysis_config["token_start"],
    #     token_end=analysis_config["token_end"],
    #     output_path=os.path.join(token_dir, f"{dataset_name}.svg")
    # )
    # logging.info(f"Saved Cosine Similarity plot to {token_dir}")
    # del frequencies4
    # torch.cuda.empty_cache()

    # 构建 activation_matrix
    # activation_matrix = build_activation_matrix(
    #     model, tokenizer, texts, matrix_config["num_layers"], matrix_config["num_experts"],
    #     generation_config["max_new_tokens"], model_config["device"]
    # )

    # plot_layer_token_similarity_bar(activation_matrix,
    #                                 output_path=os.path.join(matrix_dir, f"{dataset_name}.svg"))
    # logging.info(f"Saved activation matrix to {matrix_dir}")
    # del activation_matrix

    # 处理文本并收集频率数据
    try:
        cache_result = cache_process(
            model,
            tokenizer,
            texts,
            generation_config["max_new_tokens"],
            model_config["device"],
            generation_config.get("hot", True),
        )
    except Exception as e:
        logging.error(f"Failed to cache_process: {str(e)}", exc_info=True)
        raise

    # 将 overall_stats 保存为 JSON 文件
    try:
        # 确保 layer_idx 为字符串以符合 JSON 格式
        json_compatible_stats = {}
        for cache_type in cache_result:
            json_compatible_stats[cache_type] = {
                "total_hits": {
                    str(k): v for k, v in cache_result[cache_type]["total_hits"].items()
                },
                "total_requests": {
                    str(k): v
                    for k, v in cache_result[cache_type]["total_requests"].items()
                },
                "hit_rates": {
                    str(k): v for k, v in cache_result[cache_type]["hit_rates"].items()
                },
                "overall_hit_rate": cache_result[cache_type]["overall_hit_rate"],
            }

        # 保存到 JSON 文件
        output_path = os.path.join(cache_dir, f"{dataset_name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_compatible_stats, f, indent=4, ensure_ascii=False)
        print(f"\nSaved cache statistics to {output_path}")
    except Exception as e:
        print(f"Error saving JSON to {output_path}: {str(e)}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
