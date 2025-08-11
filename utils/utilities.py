import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from utils.properties_util import convert_standard_properties_to_tokens

def print_properties_info(age: str, gender: str, emotion: str, pitch: float, speed: float):
    """
    打印属性信息的辅助函数
    
    Args:
        age: 年龄
        gender: 性别
        emotion: 情感
        pitch: 音调
        speed: 速度
    """
    print(f'age: {age}, gender: {gender}, emotion: {emotion}, pitch: {pitch}, speed: {speed}')

def extract_embeddings_for_global_tokens(model, tokenizer, text, age: str, gender: str, emotion: str, pitch: float, speed: float,global_tokens: list = None):
    """
    提取生成全局tokens所需的embedding
    
    Args:
        model: 模型实例
        tokenizer: 分词器
        text: 输入文本
        age: 年龄
        gender: 性别
        emotion: 情感
        pitch: 音调
        speed: 速度
        global_tokens: 全局tokens
    Returns:
        torch.Tensor: 拼接后的完整embedding
    """
    device = (next(model.parameters()).device) 
    properties_tokens = convert_standard_properties_to_tokens(age, gender, emotion, pitch, speed)
    print(f'properties_tokens: {properties_tokens}')
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    properties_tokens = tokenizer.encode(properties_tokens, add_special_tokens=False)
    print(f'properties_tokens: {properties_tokens}')
    text_tokens_tensor = torch.tensor(text_tokens, dtype=torch.long, device=device)
    properties_tokens_tensor = torch.tensor(properties_tokens, dtype=torch.long, device=device)
    text_embs = model.text_embedder(text_tokens_tensor)
    properties_embs = model.text_embedder(properties_tokens_tensor)
    tag_0_emb = model.tts_tag_embedder(torch.tensor([0], dtype=torch.long, device=device))
    tag_1_emb = model.tts_tag_embedder(torch.tensor([1], dtype=torch.long, device=device))
    tag_2_emb = model.tts_tag_embedder(torch.tensor([2], dtype=torch.long, device=device))
    full_embs_for_sample = torch.cat([
        properties_embs,
        tag_2_emb, text_embs, tag_0_emb,
    ], dim=0)
    if global_tokens is not None:
        global_tokens_tensor = torch.tensor(global_tokens, dtype=torch.long, device=device)
        global_embs = model.global_embedder(global_tokens_tensor)
        full_embs_for_sample = torch.cat([
            full_embs_for_sample,
            global_embs,
            tag_1_emb
        ], dim=0)
    return full_embs_for_sample

def get_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    special_tokens = {
            'pad_token': '<|rwkv_tokenizer_end_of_text|>',
            'additional_special_tokens': [
                '<|endofprompt|>',
                '[breath]', '<strong>', '</strong>', '[noise]',
                '[laughter]', '[cough]', '[clucking]', '[accent]',
                '[quick_breath]',
                "<laughter>", "</laughter>",
                "[hissing]", "[sigh]", "[vocalized-noise]",
                "[lipsmack]", "[mn]"
            ]
        }
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer

def get_respark_tts_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    original_vocab_size = tokenizer.vocab_size
    added_tokens_file = os.path.join(os.path.dirname(__file__),'spark_tts_added_tokens.json')
    with open(added_tokens_file, 'r') as f:
        added_tokens = json.load(f)
    tokenizer.add_special_tokens(added_tokens)
    return tokenizer,original_vocab_size

def generate_global_tokens(model, tokenizer, text, age: str, gender: str, emotion: str, pitch: float, speed: float,
                           num_global_tokens: int = 4096):
    print_properties_info(age, gender, emotion, pitch, speed)
    full_embs_for_sample = extract_embeddings_for_global_tokens(model, tokenizer, text, age, gender, emotion, pitch, speed)
    device = full_embs_for_sample.device
    vocab_size = model.config.vocab_size
    eos_token_id = vocab_size - 1
    suppress_tokens = [id for id in range(num_global_tokens,vocab_size)]
    gen_args = {
        "inputs_embeds":full_embs_for_sample.unsqueeze(0),
        "attention_mask":torch.ones((1, full_embs_for_sample.shape[1]),dtype=torch.long,device=device),
        "max_new_tokens":32,
        "min_new_tokens":32,
        "do_sample":True,
        "top_k":50,
        "top_p":0.95,
        "temperature":1.0,
        "eos_token_id":eos_token_id,
        "pad_token_id":tokenizer.pad_token_id,
        "use_cache":True,
        "suppress_tokens":suppress_tokens,
        "return_dict_in_generate":True,
    }
    print(f'input_embs shape: {full_embs_for_sample.shape}')
    generated_outputs = model.generate(**gen_args)
    return generated_outputs

def generate_input_embeddings(model,tokenizer,text,global_tokens):
    device = (next(model.parameters()).device) 
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    text_tokens_tensor = torch.tensor(text_tokens, dtype=torch.long, device=device)
    text_embs = model.text_embedder(text_tokens_tensor)
    global_tokens_tensor = torch.tensor(global_tokens, dtype=torch.long, device=device)
    global_embs = model.global_embedder(global_tokens_tensor)
    tag_0_emb = model.tts_tag_embedder(torch.tensor([0], dtype=torch.long, device=device))
    tag_1_emb = model.tts_tag_embedder(torch.tensor([1], dtype=torch.long, device=device))
    tag_2_emb = model.tts_tag_embedder(torch.tensor([2], dtype=torch.long, device=device))
    input_embs = torch.cat([tag_2_emb,text_embs,tag_0_emb,global_embs,tag_1_emb],dim=0)
    return input_embs

def generate_embeddings(model, tokenizer, text, bicodec, prompt_text=None, prompt_audio=None):
    """
    为 Spark LLM 生成预测所需的输入嵌入
    
    Args:
        model: Spark LLM 模型
        tokenizer: 文本分词器
        text: 要生成语音的文本
        bicodec: BiCodecTokenizer 实例
        prompt_text: 提示文本（可选）
        prompt_audio: 提示音频数组（可选）
    
    Returns:
        dict: 包含 input_embs 的字典，用于模型预测
    """
    device = next(model.parameters()).device
    
    # 1. 处理提示音频，提取 global_tokens 和 semantic_tokens
    if prompt_audio is not None:
        # 确保音频数据是 float32 类型
        audio_data = np.array(prompt_audio, dtype=np.float32)
        target_sample_rate = bicodec.config['sample_rate']
        
        # 检查是否需要重采样
        # 注意：这里假设 prompt_audio 已经是从 soundfile 加载的，采样率信息在外部处理
        # BiCodecTokenizer 期望 16kHz 采样率的音频
        print(f"BiCodecTokenizer 期望的采样率: {target_sample_rate}Hz")
        print(f"音频数据形状: {audio_data.shape}")
        
        # 使用 BiCodec 提取 tokens (返回顺序: global_tokens, semantic_tokens)
        global_tokens, semantic_tokens = bicodec.tokenize(audio_data)
        global_tokens = global_tokens.squeeze(0).squeeze(0).detach().cpu().tolist()
        semantic_tokens = semantic_tokens.squeeze(0).squeeze(0).detach().cpu().tolist()
    else:
        global_tokens = []
        semantic_tokens = []
    
    # 2. 处理文本
    if prompt_text is not None:
        # 连接提示文本和目标文本
        full_text = prompt_text + text
        # 初始的 semantic tokens 等于 prompt_audio 提取的 semantic tokens
        initial_semantic_tokens = semantic_tokens.copy()
    else:
        full_text = text
        initial_semantic_tokens = []
    
    # 3. 获取文本 tokens
    text_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    
    # 4. 转换为张量
    text_tokens_tensor = torch.tensor(text_tokens, dtype=torch.long, device=device)
    global_tokens_tensor = torch.tensor(global_tokens, dtype=torch.long, device=device)
    semantic_tokens_tensor = torch.tensor(initial_semantic_tokens, dtype=torch.long, device=device)
    
    # 5. 获取嵌入
    text_embs = model.text_embedder(text_tokens_tensor)
    global_embs = model.global_embedder(global_tokens_tensor)
    semantic_embs = model.model.embeddings(semantic_tokens_tensor)
    
    # 6. 获取特殊标记嵌入
    tag_0_emb = model.tts_tag_embedder(torch.tensor([0], dtype=torch.long, device=device))
    tag_1_emb = model.tts_tag_embedder(torch.tensor([1], dtype=torch.long, device=device))
    tag_2_emb = model.tts_tag_embedder(torch.tensor([2], dtype=torch.long, device=device))
    
    # 7. 连接嵌入
    input_embs = torch.cat([
        tag_2_emb, 
        text_embs, 
        tag_0_emb, 
        global_embs, 
        tag_1_emb, 
        semantic_embs
    ], dim=0)
    
    # 8. 添加批次维度
    input_embs = input_embs.unsqueeze(0)  # [1, seq_len, hidden_size]
    
    return {
        "input_embs": input_embs,
        "global_tokens": global_tokens_tensor,
    }