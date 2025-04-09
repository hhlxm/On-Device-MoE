import random
import sys
import os
import logging
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

from Analyse.my_dataload import _load_data_predict
from models.DeepSeek_V2_Lite.modeling_deepseek_prefetch import DeepseekV2ForCausalLM

# 自定义数据集类
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_length=256):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]
        encoding = self.tokenizer(
            text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        labels = input_ids.clone()
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels[input_ids == self.tokenizer.pad_token_id] = -100  # 忽略 <pad>
        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}

# 设置日志
def setup_logging(output_dir, model_name, train_dataset_name, test_dataset_name):
    log_file = os.path.join(output_dir, model_name,train_dataset_name,test_dataset_name, "log.log")
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
    return model, tokenizer

# 设置模型参数
def setup_model_parameters(model):
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    predictor_params = []
    for layer in model.model.layers:
        if hasattr(layer, 'predictor') and layer.predictor is not None:
            for param in layer.predictor.parameters():
                param.requires_grad = True
            predictor_params.extend(layer.predictor.parameters())
    return predictor_params

# 加载数据集
def load_data(tokenizer, data_config):
    train_dataset = _load_data_predict(data_config["train_dataset_name"], data_type=data_config['train_type'])
    val_dataset = _load_data_predict(data_config["test_dataset_name"], data_type=data_config['test_type'])
    
    train_dataset = CustomDataset(train_dataset, tokenizer, max_length=data_config.get("max_length", 256))
    val_dataset = CustomDataset(val_dataset, tokenizer, max_length=data_config.get("max_length", 256))
    
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
    plt.savefig(save_path)
    plt.close()
    # logging.info(f"Loss plot saved to {save_path}")

# 绘制指标条形图
def plot_metrics(metric_dict, metric_name, save_path):
    if not metric_dict:
        logging.warning(f"No {metric_name} data to plot.")
        return
    
    latest_step = max(metric_dict.keys())
    latest_values = metric_dict[latest_step]
    layers = range(len(latest_values))

    plt.figure(figsize=(10, 6))
    plt.bar(layers, latest_values, color='skyblue')
    plt.xlabel("Layer Index")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"Validation {metric_name.capitalize()} per Layer at Global Step {latest_step}")
    plt.grid(True, axis='y')
    plt.savefig(save_path)
    plt.close()
    # logging.info(f"Validation {metric_name} plot saved to {save_path}")

# 主训练循环
def train(model, train_dataloader, val_dataloader, optimizer, train_config, output_config,data_config,model_config):
    scaler = GradScaler()
    train_loss, val_loss, val_iou, val_acc = {}, {}, {}, {}
    global_step = 0

    for epoch in range(train_config["num_epochs"]):
        for step, batch in enumerate(train_dataloader):
            if global_step % train_config["eval_every"] == 0:
                v_loss, v_iou, v_acc = validate(model, val_dataloader, max_steps=train_config.get("val_max_steps", 2))
                val_loss[global_step] = v_loss
                val_iou[global_step] = v_iou
                val_acc[global_step] = v_acc
                
                save_dir = os.path.join(output_config["dir"],os.path.basename(model_config["name"]).lower(),data_config['train_dataset_name'] ,data_config['test_dataset_name'])
                os.makedirs(save_dir, exist_ok=True)
                plot_losses(train_loss, val_loss, os.path.join(save_dir, "loss_plot.png"))
                plot_metrics(val_iou, "iou", os.path.join(save_dir, "val_iou_plot.png"))
                plot_metrics(val_acc, "accuracy", os.path.join(save_dir, "val_acc_plot.png"))
                
                model.train()

            train_loss[global_step] = train_step(model, batch, optimizer, scaler)

            if (step + 1) % train_config["print_every"] == 0:
                # avg_loss = train_loss[global_step]  # 单步损失
                logging.info(f"Epoch {epoch + 1}/{train_config['num_epochs']}, Step {step + 1}/{len(train_dataloader)}, "
                             f"Global Step {global_step}, Train Predictor Loss: {train_loss[global_step]:.4f}")
            global_step += 1

    logging.info("Training completed!")
    return train_loss, val_loss, val_iou, val_acc

# 设置随机种子
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# 主函数
def main():
    # 默认配置文件路径
    config_path = sys.argv[1] if len(sys.argv) > 1 else "/home/pairshoe/lxm/train_config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found. Please provide a valid YAML config file.")
    
    # 加载配置
    config = load_config(config_path)
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    train_config = config.get("train", {})
    output_config = config.get("output", {})

    # 设置日志
    setup_logging(
        output_config["dir"], 
        os.path.basename(model_config["name"]).lower(), 
        data_config["train_dataset_name"], 
        data_config["test_dataset_name"]
    )
    logging.info(f"Running with config: {yaml.dump(config)}")

    # 设置随机种子
    set_random_seed(config.get("seed", 2025))

    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_config)

    # 设置模型参数
    predictor_params = setup_model_parameters(model)

    # 设置优化器
    optimizer = torch.optim.Adam(
        predictor_params, 
        lr=train_config['optimizer']['lr'], 
        eps=train_config['optimizer']['eps'],
    )

    # 加载数据
    train_dataloader, val_dataloader = load_data(tokenizer, data_config)

    # 训练
    train(model, train_dataloader, val_dataloader, optimizer, train_config, output_config,data_config,model_config)

if __name__ == "__main__":
    main()