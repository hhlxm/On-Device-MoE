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


from util import setup_logging, load_config, set_random_seed, plot_losses, plot_metrics, load_data, load_model_and_tokenizer
import statistics
import math
from huggingface_hub import login
login("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")



# 设置模型参数
def setup_model_parameters(model):
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    predictor_params = []
    for layer in model.model.layers:
        if hasattr(layer, 'predictor') and layer.predictor is not None:
            for name, param in layer.predictor.named_parameters():
                # 如果是 lm_head.weight，则不启用梯度
                if name == "lm_head.weight":
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    predictor_params.append(param)
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
def train_step(model, batch, optimizer,scaler):
    input_ids = batch['input_ids'].to("cuda")
    labels = batch['labels'].to("cuda")
    attention_mask = batch['attention_mask'].to("cuda")
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()

    # """
    # 执行一个训练步骤（全精度训练）
    
    # 参数:
    #     model: 模型
    #     batch: 当前批次的数据，包含 input_ids, labels, attention_mask
    #     optimizer: 优化器
    # """
    # # 将数据移动到 GPU
    # input_ids = batch['input_ids'].to("cuda")
    # labels = batch['labels'].to("cuda")
    # attention_mask = batch['attention_mask'].to("cuda")

    # # 前向传播
    # outputs = model(
    #     input_ids=input_ids,
    #     labels=labels,
    #     attention_mask=attention_mask
    # )
    # loss = outputs.loss

    # # 反向传播
    # optimizer.zero_grad()
    # loss.backward()

    # # 参数更新
    # optimizer.step()

    # # 返回损失值
    # return loss.item()


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
    scaler = torch.amp.GradScaler()
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
                plot_losses(train_loss, val_loss, os.path.join(save_dir, "loss_plot.png"))
                plot_losses(
                    {key: np.mean(values) if values else None for key, values in val_acc.items()}
                    , {key: np.mean(values) if values else None for key, values in val_iou.items()}, os.path.join(save_dir, "val_acc.png"))
                plot_metrics(val_iou, "iou", os.path.join(save_dir, "val_iou_plot.png"))
                plot_metrics(val_acc, "accuracy", os.path.join(save_dir, "val_acc_plot.png"))
                logging.info(f"Validation acc: {statistics.mean(v_acc):.7f}")
                model.train()

            train_loss[global_step] = train_step(model, batch, optimizer, scaler)
            # scheduler.step()
            if (step + 1) % train_config["print_every"] == 0:
                # avg_loss = train_loss[global_step]  # 单步损失
                logging.info(f"Epoch {epoch + 1}/{train_config['num_epochs']}, Step {step + 1}/{len(train_dataloader)}, "
                             f"Global Step {global_step}, Train Predictor Loss: {train_loss[global_step]:.7f} ")
            global_step += 1
            # if model_config.get("type","pregate") in ["prefetch","pretoken"] and global_step%100==0:
            #     save_model(model, train_config,save_dir)
            
    if model_config.get("type","pregate") in ["prefetch","pretoken","pregate"]:
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
        # checkpoint = torch.load("/home/fit/renju/WORK/lxm/Predict/results/DeepSeek_V2_Lite/output_pregate_instruct_finetune_linear/deepseek_v2_lite/alpaca/alpaca/checkpoints/checkpoint_epoch_1.pt", map_location="cuda")
        # print(checkpoint['model_state_dict'].keys())
        # model.load_state_dict(checkpoint['model_state_dict'], strict=False) 
        # model.save_pretrained("/home/fit/renju/WORK/lxm/models/test", safe_serialization=True)
        # tokenizer.save_pretrained("/home/fit/renju/WORK/lxm/models/test")
        # return
        if model_config.get("type","pregate") in ["pretoken","pregate","prefetch"]:
        #     from models.DeepSeek_V2_Lite.modeling_deepseek_pretoken import DeepseekV2MoE
            for layer in model.model.layers:  # 假设 decoder 层位于 model.model.layers 中
                if layer.predictor is not None:
                    # print("init")
                    nn.init.kaiming_uniform_(layer.predictor.linear.weight, a=math.sqrt(5))
                    # nn.init.kaiming_uniform_(layer.predictor.linear[2].weight, a=math.sqrt(5))

                    # layer.predictor.linear.weight = nn.Parameter(layer.mlp.gate.weight.float())
                        
    elif train_config['train_max_steps'] == -1: #only eval
        if model_config.get("type","pregate") in ["prefetch","pretoken","pregate"]:
            load_checkpoint(model, train_config['checkpoint_path'])
        v_loss, v_iou, v_acc = validate(model, val_dataloader, max_steps=train_config.get("val_max_steps", 2))
        plot_metrics(v_iou, "iou", os.path.join(save_dir,data_config['test_dataset_name'],"figures", "val_iou_plot.png"),save_data=True)
        plot_metrics(v_acc, "accuracy", os.path.join(save_dir,data_config['test_dataset_name'],"figures", "val_acc_plot.png"),save_data=True)
        logging.info("Validation completed!")
        return
        
    # 设置模型参数
    predictor_params = setup_model_parameters(model)

    # 设置优化器
    optimizer = torch.optim.Adam(
        predictor_params, 
        lr=train_config['optimizer']['lr'], 
        eps = train_config['optimizer']['eps'],
    )
    # base_lr = train_config['optimizer']['lr']
    # total_steps = train_config['train_max_steps']
    # warmup_steps = train_config.get("warmup_steps", int(0.1 * total_steps))  # 默认10%预热
    # min_lr = train_config.get("min_lr", 0.0)
    # def lr_lambda(current_step):
    #     if current_step < warmup_steps:
    #         return current_step / max(1, warmup_steps)
    #     progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    #     cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    #     return max(min_lr / base_lr, cosine_decay)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    
    # 训练
    train(model, train_dataloader, val_dataloader, optimizer, train_config, data_config,model_config,save_dir)

if __name__ == "__main__":
    main()