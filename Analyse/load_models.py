'''
Author: hhlxm 578723542@qq.com
Date: 2025-04-11 14:36:20
LastEditors: hhlxm 578723542@qq.com
LastEditTime: 2025-07-01 00:33:31
FilePath: /lxm/Analyse/load_models.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import sys
import os
# 添加当前目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch import float16


def load_model_and_tokenizer(model_config, device, torch_dtype=float16):
    """加载模型和分词器"""
    model_name = model_config["name"]
    model_type = model_config["type"] 
    
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if 'cache_ratio' in config.to_dict():
        config.cache_ratio = model_config.get("cache_ratio", 0)
        print(f"cache_ratio: {config.cache_ratio}")
    if 'pre_ratio' in config.to_dict():
        config.pre_ratio = model_config.get("pre_ratio", 1.0)
        print(f"pre_ratio: {config.pre_ratio}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # 设置 pad_token 为 eos_token

    # 根据模型类型动态加载
    if model_type == "DeepseekV2":
        from models.DeepSeek_V2_Lite.modeling_deepseek_pretoken_cache_w_load import DeepseekV2ForCausalLM as ModelClass
        model = ModelClass.from_pretrained(
        model_name,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch_dtype,
        device_map="auto",  # 自动分配到多个 GPU
          # 自动分配到多个 GPU
    )
        checkpoint = torch.load("/home/fit/renju/WORK/lxm/Predict/results/DeepSeek_V2_Lite/output_pretoken_instruct_finetune/deepseek_v2_lite/alpaca/alpaca/checkpoints/checkpoint_epoch_1.pt", map_location="cuda")
        print(checkpoint['model_state_dict'].keys())
        model.load_state_dict(checkpoint['model_state_dict'], strict=False) 
    elif model_type == "Qwen2Moe":
        from models.Qwen1_5_MoE_A2_7B.modeling_qwen2_moe_pretoken_cache import Qwen2MoeForCausalLM as ModelClass
        model = ModelClass.from_pretrained(
        model_name,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch_dtype,
        device_map="auto",  # 自动分配到多个 GPU
         # 自动分配到多个 GPU
    )
        checkpoint = torch.load("/home/fit/renju/WORK/lxm/Predict/results/Qwen1_5_MoE_A2_7B/output_pretoken_instruct_finetune/qwen1_5_moe_a2_7b/alpaca/alpaca/checkpoints/checkpoint_epoch_1.pt", map_location="cuda")
        print(checkpoint['model_state_dict'].keys())
        model.load_state_dict(checkpoint['model_state_dict'], strict=False) 
    elif model_type == "PhiMoE":
        from models.Phi_3_5_MoE_instruct.modeling_phimoe_pretoken_cache import PhiMoEForCausalLM as ModelClass
        model = ModelClass.from_pretrained(
        model_name,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch_dtype,
        device_map="auto",  # 自动分配到多个 GPU
    )
        checkpoint = torch.load("/home/fit/renju/WORK/lxm/Predict/results/Phi_3_5_MoE_instruct/output_pretoken_instruct_finetune/phi_3_5_moe_instruct/alpaca/alpaca/checkpoints/checkpoint_epoch_1.pt", map_location="cuda")
        print(checkpoint['model_state_dict'].keys())
        model.load_state_dict(checkpoint['model_state_dict'], strict=False) 
    elif model_type == "Mixtral":
        from models.Mixtral_8x7B_v0_1.modeling_mixtral_pretoken_cache import MixtralForCausalLM as ModelClass
        model = ModelClass.from_pretrained(
        model_name,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch_dtype,
        device_map="auto",  # 自动分配到多个 GPU
         # 自动分配到多个 GPU
    )
        checkpoint = torch.load("/home/fit/renju/WORK/lxm/Predict/results/Mixtral_8x7B_v0_1/output_pretoken_instruct_finetune/mixtral_8x7b_v0_1/alpaca/alpaca/checkpoints/checkpoint_epoch_1.pt", map_location="cuda")
        print(checkpoint['model_state_dict'].keys())
        model.load_state_dict(checkpoint['model_state_dict'], strict=False) 
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


    
    return model, tokenizer, config