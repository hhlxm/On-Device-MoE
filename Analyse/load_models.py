import sys
import os
# 添加当前目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch import float16


def load_model_and_tokenizer(model_config, device, torch_dtype=float16):
    """加载模型和分词器"""
    model_name = model_config["name"]
    model_type = model_config["type"] 
    
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # 设置 pad_token 为 eos_token

    # 根据模型类型动态加载
    if model_type == "DeepseekV2":
        from models.DeepSeek_V2_Lite.modeling_deepseek import DeepseekV2ForCausalLM as ModelClass
        model = ModelClass.from_pretrained(
        model_name,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch_dtype,
          # 自动分配到多个 GPU
    ).to(device)
    elif model_type == "Qwen2Moe":
        from models.Qwen1_5_MoE_A2_7B.modeling_qwen2_moe import Qwen2MoeForCausalLM as ModelClass
        model = ModelClass.from_pretrained(
        model_name,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch_dtype,
         # 自动分配到多个 GPU
    ).to(device)
    elif model_type == "PhiMoE":
        from models.Phi_3_5_MoE_instruct.modeling_phimoe import PhiMoEForCausalLM as ModelClass
        model = ModelClass.from_pretrained(
        model_name,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch_dtype,
        device_map="auto",  # 自动分配到多个 GPU
    )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


    
    return model, tokenizer, config