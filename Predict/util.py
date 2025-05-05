import json
from pathlib import Path
import random
import shutil
import sys
import os
import logging
from tqdm import tqdm
import yaml
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import float16
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
# 添加当前目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Analyse.my_dataload import CustomDataset



# 加载模型和分词器
def load_model_and_tokenizer(model_config):
    type = model_config['type']
    if type not in ['pregate','pretoken','prefetch']:
        raise ValueError("Invalid type. Choose from 'pregate', 'finetune', or 'pretoken'.")
    
    config = AutoConfig.from_pretrained(model_config["name"], trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    
    if type == 'pregate':
        from models.DeepSeek_V2_Lite.modeling_deepseek_pregate_test import DeepseekV2ForCausalLM
        from models.Phi_3_5_MoE_instruct.modeling_phimoe_pregate_test import PhiMoEForCausalLM
        from models.Qwen1_5_MoE_A2_7B.modeling_qwen2_moe_pregate_test import Qwen2MoeForCausalLM
        from models.Mixtral_8x7B_v0_1.modeling_mixtral_pregate_test import MixtralForCausalLM
    elif type == 'pretoken':
        from models.DeepSeek_V2_Lite.modeling_deepseek_pretoken import DeepseekV2ForCausalLM
        from models.Phi_3_5_MoE_instruct.modeling_phimoe_pretoken import PhiMoEForCausalLM
    elif type == 'prefetch':
        from models.DeepSeek_V2_Lite.modeling_deepseek_prefetch import DeepseekV2ForCausalLM
        
    
        
    if 'deepseek' in model_config["name"].lower():
        model = DeepseekV2ForCausalLM.from_pretrained(
            model_config["name"], trust_remote_code=True, config=config, 
            torch_dtype=float16, device_map=model_config.get("device_map", "auto")
        )
    elif 'phi' in model_config["name"].lower():
        model = PhiMoEForCausalLM.from_pretrained(
            model_config["name"], trust_remote_code=True, config=config, 
            torch_dtype=float16, device_map=model_config.get("device_map", "auto")
        )
    elif 'qwen' in model_config["name"].lower():
        model = Qwen2MoeForCausalLM.from_pretrained(
            model_config["name"], trust_remote_code=True, config=config, 
            torch_dtype=float16, device_map=model_config.get("device_map", "auto")
        )
    elif 'mixtral' in model_config["name"].lower():
        model = MixtralForCausalLM.from_pretrained(
            model_config["name"], trust_remote_code=True, config=config, 
            torch_dtype=float16, device_map=model_config.get("device_map", "auto")
        )
    if type == 'pregate':
        model.model.pre_dis=model_config.get("pre_dis", 1 if 'deepseek' in model_config["name"].lower() else 0)
        model.model.pre_ahead=model_config.get("pre_ahead", 1)
        print(f"Model loaded from {model_config['name']} with type {type} Pre_dis:{model.model.pre_dis}  Pre_ahead:{model.model.pre_ahead}")
    return model, tokenizer



# 设置日志
def setup_logging(save_dir, test_dataset_name):
    log_file = os.path.join(save_dir,test_dataset_name, "log.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

# 加载 YAML 配置
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config





# 加载数据集
def load_data(tokenizer, data_config):

    train_dataset = CustomDataset(data_config["train_dataset_name"], data_config['train_type'], tokenizer, max_length=data_config.get("max_length", 256))
    val_dataset = CustomDataset(data_config["test_dataset_name"], data_config['test_type'], tokenizer, max_length=data_config.get("max_length", 256))
    train_dataloader = DataLoader(
        train_dataset, batch_size=data_config["batch_size"], shuffle= True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=data_config["batch_size"], shuffle= True
    )
    
    return train_dataloader, val_dataloader



# 绘制损失曲线
def plot_losses(train_loss, val_loss, save_path):
    train_steps = list(train_loss.keys())
    train_values = list(train_loss.values())
    val_steps = list(val_loss.keys())
    val_values = list(val_loss.values())

    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_values, label="Train Loss", marker='o')
    plt.plot(val_steps, val_values, label="Validation Loss", marker='x')
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path,format='svg')
    plt.close()
    # logging.info(f"Loss plot saved to {save_path}")

# 绘制指标条形图
def plot_metrics(metric_dict, metric_name, save_path,save_data=False):
    if not metric_dict:
        logging.warning(f"No {metric_name} data to plot.")
        return
    
    if isinstance(metric_dict,dict):
        latest_step = max(metric_dict.keys())
        latest_values = metric_dict[latest_step]
    else:
        latest_values = metric_dict
        
    layers = range(len(latest_values))

    plt.figure(figsize=(10, 6))
    plt.bar(layers, latest_values, color='skyblue')
    plt.xlabel("Layer Index")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"Validation {metric_name.capitalize()} per Layer ")
    plt.grid(True, axis='y')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path,format='svg')

    
    if save_data:
        # 构造 JSON 文件路径
        json_path = Path(save_path).with_suffix('.json')
        
        # 保存 latest_values 数据为 JSON 文件
        with open(json_path, 'w') as f:
            json.dump({
                "layers": list(layers),
                "values": latest_values if isinstance(latest_values, list) else latest_values.tolist(),
                "metric_name": metric_name,
            }, f, indent=4)
        logging.info(f"Metrics data saved to {json_path}")
    
    
    plt.close()
    
    
    
# 设置随机种子
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)