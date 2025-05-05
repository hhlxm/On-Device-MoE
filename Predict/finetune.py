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
from datasets import load_dataset

# 添加当前目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from Analyse.my_dataload import CustomDataset
from models.DeepSeek_V2_Lite.modeling_deepseek_pregate import DeepseekV2ForCausalLM

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, split, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset("/home/fit/renju/WORK/lxm/datasets/alpaca")[split]
        self.prompt_column = "instruction"
        self.input_column = "input"
        self.response_column = "output"
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        max_source_length = self.max_length // 2
        max_target_length = self.max_length // 2
        
        query = example[self.prompt_column] + example[self.input_column]
        answer = example[self.response_column]
        
        # 直接使用 query，不构建 prompt
        a_ids = self.tokenizer.encode(
            text=query,
            add_special_tokens=True,
            truncation=True,
            max_length=max_source_length
        )
        b_ids = self.tokenizer.encode(
            text=answer,
            add_special_tokens=False,
            truncation=True,
            max_length=max_target_length
        )
        
        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
        labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]
        
        pad_len = self.max_length - len(input_ids) + 1
        
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [self.tokenizer.pad_token_id] * pad_len
        labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]
        
        attention_mask = [1] * len(a_ids + b_ids + [self.tokenizer.eos_token_id]) + [0] * pad_len
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

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

# 加载模型和分词器
def load_model_and_tokenizer(model_config):
    config = AutoConfig.from_pretrained(model_config["name"], trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = DeepseekV2ForCausalLM.from_pretrained(
        model_config["name"], trust_remote_code=True, config=config, 
        torch_dtype=float16, device_map=model_config.get("device_map", "auto")
    )
    model.model.pre_dis = model_config.get("pre_dis", 1)
    print("pre dis:", model.model.pre_dis)
    return model, tokenizer

# 设置模型参数
def setup_model_parameters(model):
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    predictor_params = []
    for layer in model.model.layers:
        if hasattr(layer.mlp, 'gate') and layer.mlp.gate is not None:
            for param in layer.mlp.gate.parameters():
                param.requires_grad = True
            predictor_params.extend(layer.mlp.gate.parameters())
    return predictor_params

def save_model(model, train_config, save_dir,step):
    save_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    
    predictor_state_dict = {k: v for k, v in model.state_dict().items() if 'gate' in k and 'gate_proj' not in k}
    
    checkpoint = {
        'model_state_dict': predictor_state_dict,
    }
    
    save_path = os.path.join(save_dir, f"checkpoint_step_{step}.pt")
    torch.save(checkpoint, save_path)
    logging.info(f"Model checkpoint saved to {save_path}")

# 加载检查点
def load_checkpoint(model, checkpoint_path, map_location="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # 只加载 predictor 参数
    
    logging.info(f"Loaded checkpoint from {checkpoint_path}")
    
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

# 训练一个 batch
def train_step(model, batch, optimizer, scaler):
    input_ids = batch['input_ids'].to("cuda")
    labels = batch['labels'].to("cuda")
    attention_mask = batch['attention_mask'].to("cuda")
    with autocast():
        outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()

# 验证过程
def validate(model, val_dataloader, max_steps=2):
    model.eval()
    v_loss, v_iou, v_acc, v_tokens = 0, [], [], []
    with torch.no_grad():
        for val_step, val_batch in enumerate(val_dataloader):
            if val_step >= max_steps:
                break
            val_input_ids = val_batch['input_ids'].to("cuda")
            val_labels = val_batch['labels'].to("cuda")
            val_attention_mask = val_batch['attention_mask'].to("cuda")
            val_outputs = model(input_ids=val_input_ids, labels=val_labels, attention_mask=val_attention_mask)
            v_loss += val_outputs.loss.item()
            v_iou.append(model.model.iou_scores)
            v_acc.append(model.model.accuracy_scores)
            v_tokens.append(model.model.tokens)
            
    avg_loss = v_loss / val_step if val_step > 0 else 0
    avg_iou = [sum(step_iou[i] for step_iou in v_iou) / val_step for i in range(len(v_iou[0]))] if v_iou else []
    avg_acc = [sum(v_acc[s][i] * v_tokens[s] for s in range(val_step)) / sum(v_tokens) for i in range(len(v_acc[0]))] if v_acc and v_tokens else []
    return avg_loss, avg_iou, avg_acc

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


    

# 主训练循环
def train(model, train_dataloader, val_dataloader, optimizer, train_config, data_config,save_dir):
    scaler = GradScaler()
    train_loss, val_loss, val_iou, val_acc = {}, {}, {}, {}
    global_step = 0
    save_dir = os.path.join(save_dir ,data_config['test_dataset_name'])
    for epoch in range(train_config["num_epochs"]):
        for step, batch in enumerate(train_dataloader):
            if global_step >= train_config["train_max_steps"]:
                break
            if global_step % train_config["eval_every"] == 0:
                v_loss, v_iou, v_acc = validate(model, val_dataloader, max_steps=train_config.get("val_max_steps", 2))
                val_loss[global_step] = v_loss
                val_iou[global_step] = v_iou
                val_acc[global_step] = v_acc
                
                
                os.makedirs(save_dir, exist_ok=True)
                plot_losses(train_loss, val_loss, os.path.join(save_dir, "loss_plot.svg"))
                plot_metrics(val_iou, "iou", os.path.join(save_dir, "val_iou_plot.svg"))
                plot_metrics(val_acc, "accuracy", os.path.join(save_dir, "val_acc_plot.svg"))
                
                model.train()

            train_loss[global_step] = train_step(model, batch, optimizer, scaler)

            if (step + 1) % train_config["print_every"] == 0:
                # avg_loss = train_loss[global_step]  # 单步损失
                logging.info(f"Epoch {epoch + 1}/{train_config['num_epochs']}, Step {step + 1}/{len(train_dataloader)}, "
                             f"Global Step {global_step}, Train Predictor Loss: {train_loss[global_step]:.4f}")
            global_step += 1
            
            if global_step%100 == 0 :
                save_model(model, train_config,save_dir,global_step)
            
    
    save_model(model, train_config,save_dir,global_step)
    logging.info("Training completed!")
    return val_loss

# 设置随机种子
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# 主函数
def main():
    # 默认配置文件路径
    config_path = sys.argv[1] if len(sys.argv) > 1 else "/home/fit/renju/WORK/lxm/train_config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found. Please provide a valid YAML config file.")
    
    # 加载配置
    config = load_config(config_path)
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    train_config = config.get("train", {})
    output_config = config.get("output", {})
    
    if (train_config['num_epochs']==0) and (train_config['train_max_steps']==-1):
        save_dir = os.path.join(output_config["dir"], os.path.basename(model_config["name"]).lower(), 
                            data_config['train_dataset_name'],'test')
    else:
        save_dir = os.path.join(output_config["dir"], os.path.basename(model_config["name"]).lower(), 
                            data_config['train_dataset_name'])

    # 设置日志
    setup_logging(
        save_dir, 
        data_config["test_dataset_name"]
    )
    logging.info(f"Running with config: {yaml.dump(config)}")

    # 设置随机种子
    set_random_seed(config.get("seed", 2025))

    config_save_path = os.path.join(save_dir,data_config['test_dataset_name'], f"config_epoch_{train_config['num_epochs']}.yaml")
    shutil.copy(config_path, config_save_path)
    logging.info(f"YAML config saved to {config_save_path}")
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_config)

    # 加载数据
    train_dataloader, val_dataloader = load_data(tokenizer, data_config)

    if (not train_config['num_epochs']==0) and (train_config['train_max_steps']==-1):
        train_config['train_max_steps'] = len(train_dataloader) * train_config['num_epochs']
        logging.info(f"train_max_steps set to {train_config['train_max_steps']}")
    elif train_config['train_max_steps'] == -1: #only eval
        load_checkpoint(model, train_config['checkpoint_path'])
        v_loss, v_iou, v_acc = validate(model, val_dataloader, max_steps=train_config.get("val_max_steps", 2))
        plot_metrics(v_iou, "iou", os.path.join(save_dir,data_config['test_dataset_name'],"figures", "val_iou_plot.svg"),save_data=True)
        plot_metrics(v_acc, "accuracy", os.path.join(save_dir,data_config['test_dataset_name'],"figures", "val_acc_plot.svg"),save_data=True)
        logging.info("Validation completed!")
        return
        
    # 设置模型参数
    predictor_params = setup_model_parameters(model)

    # 设置优化器
    optimizer = torch.optim.Adam(
        predictor_params, 
        lr=train_config['optimizer']['lr'], 
        eps=train_config['optimizer']['eps'],
    )

    # 训练
    train(model, train_dataloader, val_dataloader, optimizer, train_config, data_config,save_dir)

if __name__ == "__main__":
    main()