import logging
import os
import shutil
import numpy as np
import yaml
import random
import torch
from torch import float16
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
import copy
import sys

# 假设这些函数定义在 utli.py 中
from load_models import load_model_and_tokenizer
from utli import draw, draw_batch, plot_cosine_similarity_avg_cross_batch, plot_layer_sums, plot_layer_token_similarity_bar
from my_dataload import load_dataset_sample

def setup_logging(output_dir, model_short_name,dataset_name):
    """设置日志，保存到文件并输出到控制台"""
    log_file = os.path.join(output_dir, model_short_name,dataset_name,f"log.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # 保存到文件
            logging.StreamHandler(sys.stdout)  # 输出到控制台
        ]
    )

def load_config(config_path):
    """读取 YAML 配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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
    return frequencies1,frequencies2, frequencies3, frequencies4

def build_activation_matrix(model, tokenizer, texts, num_layers, num_experts, max_new_tokens, device):
    model.reset_all_expert_counts()
    model.reset_all_expert_hit_rate()
    model.reset_all_token_frequency()
    """构建 activation_matrix 并处理文本"""
    activation_matrix = np.zeros((len(texts), num_layers, num_experts))

    for text_idx, input_text in enumerate(tqdm(texts, desc="Building activation matrix")):
        print(f"text_idx: {text_idx}")
        if not input_text.strip():  # 跳过空文本
            continue

        model.reset_all_expert_counts()

        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
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
                activation_matrix[text_idx, layer_counter, expert_idx] = frequencies1[layer_idx]["routed"][expert_idx]
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
    config_path = sys.argv[1] if len(sys.argv) > 1 else "/home/fit/renju/WORK/lxm/Analyse/config/config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found. Please provide a valid YAML config file.")
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
    request_dir = os.path.join(output_config["dir"], model_short_name,dataset_name, "Request_level")
    cache_dir = os.path.join(output_config["dir"], model_short_name,dataset_name, "Cache_hit")
    token_dir = os.path.join(output_config["dir"], model_short_name,dataset_name, "Token_level")
    matrix_dir = os.path.join(output_config["dir"], model_short_name,dataset_name, "Sequence_level")
    
    
    # 确保所有目录存在
    for dir_path in [request_dir, cache_dir, token_dir, matrix_dir]:
        os.makedirs(dir_path, exist_ok=True)

    setup_logging(output_config["dir"], model_short_name,dataset_name)
    logging.info(f"Running with config: {yaml.dump(config)}")

    # 复制配置文件到输出目录
    config_output_path = os.path.join(output_config["dir"],model_short_name,dataset_name, f"config.yaml")
    shutil.copy(config_path, config_output_path)

    # 加载模型和分词器
    model, tokenizer, _ = load_model_and_tokenizer(model_config, model_config.get("device", "cuda"))

    # 加载数据集
    texts = load_dataset_sample(dataset_name, dataset_config["sample_size"])

    # 处理文本并收集频率数据
    try:
        frequencies1,frequencies2, frequencies3, frequencies4 = process_texts(
            model, tokenizer, texts, generation_config["max_new_tokens"], model_config["device"]
        )
    except Exception as e:
        logging.error(f"Failed to process texts: {str(e)}", exc_info=True)
        raise

    # 绘制并保存图表
    draw(
        frequencies1,
        title="Expert Activation Heatmap",
        output_path=os.path.join(request_dir, f"{dataset_name}.svg")
    )
    logging.info(f"Saved Expert Activation Heatmap to {request_dir}")
    del frequencies1
    torch.cuda.empty_cache()
    
    
    plot_layer_sums(
        frequencies2,
        title="Layer Top_sum Average",
        output_path=os.path.join(matrix_dir, f"{dataset_name}.svg")
    )
    logging.info(f"Saved Layer Top_sum Average to {matrix_dir}")
    del frequencies2
    torch.cuda.empty_cache()
    
    draw_batch(
        frequencies3,
        title="Cache Hit Rate",
        output_path=os.path.join(cache_dir, f"{dataset_name}.svg")
    )
    logging.info(f"Saved Cache Hit Rate to {cache_dir}")
    del frequencies3
    torch.cuda.empty_cache()
    
    plot_cosine_similarity_avg_cross_batch(
        frequencies4,
        layers=range(analysis_config["layer_start"], analysis_config["layer_end"]),
        token_start=analysis_config["token_start"],
        token_end=analysis_config["token_end"],
        output_path=os.path.join(token_dir, f"{dataset_name}.svg")
    )
    logging.info(f"Saved Cosine Similarity plot to {token_dir}")
    del frequencies4
    torch.cuda.empty_cache()



    # 构建 activation_matrix
    # activation_matrix = build_activation_matrix(
    #     model, tokenizer, texts, matrix_config["num_layers"], matrix_config["num_experts"],
    #     generation_config["max_new_tokens"], model_config["device"]
    # )

    # plot_layer_token_similarity_bar(activation_matrix,
    #                                 output_path=os.path.join(matrix_dir, f"{dataset_name}.svg"))
    # logging.info(f"Saved activation matrix to {matrix_dir}")
    # del activation_matrix
    del model
    torch.cuda.empty_cache()
    

if __name__ == "__main__":
    main()