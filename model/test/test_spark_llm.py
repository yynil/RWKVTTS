from model.llm.spark_llm import RWKV7SpeechConfig,RWKV7ForSpeech

import os
from safetensors.torch import load_file
import glob

def load_multiple_safetensors(model_path):
    """加载多个 safetensors 文件并合并它们
    
    Args:
        model_path: 包含 safetensors 文件的目录路径
        
    Returns:
        合并后的 state dict
    """
    # 获取所有 safetensors 文件
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {model_path}")
    
    # 创建合并的 state dict
    merged_state_dict = {}
    
    # 加载并合并每个文件
    for file_path in safetensors_files:
        print(f"Loading {file_path}")
        state_dict = load_file(file_path)
        merged_state_dict.update(state_dict)
    
    return merged_state_dict

file_path = os.path.join(os.path.dirname(__file__), "audio_rwkv.config")
config = RWKV7SpeechConfig.from_pretrained(file_path)
print(config)
# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_config(config,trust_remote_code=True)
# print(model)

model = RWKV7ForSpeech._from_config(config)
print(model)
model_path = '/home/yueyulin/models/rwkv7-0.1B-g1/'
state_dict = load_multiple_safetensors(model_path)
model.copy_state_dict(state_dict)

