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
import torch.nn as nn
# 添加当前目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Analyse.my_dataload import CustomDataset
from util import setup_logging, load_config, set_random_seed, plot_losses, plot_metrics, load_data, load_model_and_tokenizer


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

def save_model(model, train_config, save_dir):
    save_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    
    predictor_state_dict = {k: v for k, v in model.state_dict().items() if 'predictor' in k}
    
    checkpoint = {
        'model_state_dict': predictor_state_dict,
    }
    
    save_path = os.path.join(save_dir, f"checkpoint_epoch_{train_config['num_epochs']}.pt")
    torch.save(checkpoint, save_path)
    logging.info(f"Model checkpoint saved to {save_path}")

# 加载检查点
def load_checkpoint(model, checkpoint_path, map_location="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # 只加载 predictor 参数
    
    logging.info(f"Loaded checkpoint from {checkpoint_path}")
    

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



    

# 主训练循环
def train(model, train_dataloader, val_dataloader, optimizer, train_config, data_config,model_config,save_dir):
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
            
    if model_config.get("type","pregate") in ["prefetch","pretoken"]:
        save_model(model, train_config,save_dir)
    logging.info("Training completed!")
    return val_loss



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
        
        # if model_config.get("type","pregate") in ["pretoken"]:
        #     from models.DeepSeek_V2_Lite.modeling_deepseek_pretoken import DeepseekV2MoE
        #     for layer in model.model.layers:  # 假设 decoder 层位于 model.model.layers 中
        #         if isinstance(layer.mlp, DeepseekV2MoE):
        #                 # 获取 MoE 中的 gate 的 weight
        #                 gate_weight = layer.mlp.gate.weight
        #                 # 将 predictor 的 weight 替换为 gate 的 weight
        #                 layer.predictor.weight = nn.Parameter(gate_weight.float())
                        
    elif train_config['train_max_steps'] == -1: #only eval
        if model_config.get("type","pregate") in ["prefetch","pretoken"]:
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
    )

    # 训练
    train(model, train_dataloader, val_dataloader, optimizer, train_config, data_config,model_config,save_dir)

if __name__ == "__main__":
    main()