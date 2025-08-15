import click
from webrwkv_py import Model, ThreadRuntime, get_available_adapters_py
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from transformers import AutoTokenizer
import torch
import numpy as np
import cupy as cp
import cupyx.scipy as cupy_scipy
from torch.nn import functional as F
from math import inf
from tn.chinese.normalizer import Normalizer as ChineseNormalizer
from tn.english.normalizer import Normalizer as EnglishNormalizer
chinese_normalizer = ChineseNormalizer(remove_erhua=False, full_to_half=False, overwrite_cache=False, remove_interjections=False)
english_normalizer = EnglishNormalizer()
from utils.utilities import generate_global_tokens,generate_input_embeddings,extract_embeddings_for_global_tokens,convert_standard_properties_to_tokens
import re
import os

try:
    import questionary
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False
    print("⚠️  提示: 安装 'questionary' 库可以获得更好的交互式体验")
    print("    pip install questionary")
def detect_token_lang(token: str) -> str:
    """基于字符集合的简单词级语言检测。返回 'en' 或 'zh'。
    - 含有中文字符则判为 'zh'
    - 含有英文字母则判为 'en'
    - 两者都有时优先中文（更适合中文句子里的中英混排）
    - 都没有则回退为 'en'
    """
    if not token:
        return 'en'
    has_zh = re.search(r"[\u4e00-\u9fff]", token) is not None
    has_en = re.search(r"[A-Za-z]", token) is not None
    if has_zh and not has_en:
        return 'zh'
    if has_en and not has_zh:
        return 'en'
    if has_zh and has_en:
        return 'zh'
    return 'en'

def sample_logits(logits, temperature=1.0, top_p=0.85, top_k=0):
    if temperature == 0:
        temperature = 1.0
        top_p = 0
    
    # 将 List[float] 转换为 cupy 数组
    if isinstance(logits, list):
        logits = cp.array(logits)
    
    # 使用 cupy 的 softmax
    probs = cupy_scipy.special.softmax(logits, axis=-1)
    top_k = int(top_k)
    

    
    # 使用 cupy 进行采样
    sorted_ids = cp.argsort(probs)
    sorted_probs = probs[sorted_ids][::-1]
    cumulative_probs = cp.cumsum(sorted_probs)
    
    # 找到 top_p 的截止点
    cutoff_mask = cumulative_probs >= top_p
    if cp.any(cutoff_mask):
        cutoff_idx = cp.argmax(cutoff_mask)
        cutoff = float(sorted_probs[cutoff_idx])
        probs[probs < cutoff] = 0
    
    # 应用 top_k 过滤
    if top_k < len(probs) and top_k > 0:
        probs[sorted_ids[:-top_k]] = 0
    
    # 应用温度
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)
    
    # 重新归一化概率
    probs = probs / cp.sum(probs)
    
    # 使用 cupy 的随机选择
    out = cp.random.choice(a=len(probs), size=1, p=probs)
    return int(out[0])



@click.command()
@click.option("--model_path", type=str, required=True)
@click.option("--text", type=str, required=True)
@click.option("--age", type=str, required=False,default='middle-aged')
@click.option("--gender", type=str, required=False,default='female')
@click.option("--emotion", type=str, required=False,default='neutral')
@click.option("--pitch", type=str, required=False,default='medium_pitch')
@click.option("--speed", type=str, required=False,default='medium')
@click.option("--device", type=str, required=False,default='cuda:0')
@click.option("--audio_tokenizer_path", type=str, required=False,default="/home/yueyulin/models/Spark-TTS-0.5B/")
@click.option("--output_path", type=str, required=False,default="generated_from_webrwkv.wav")
@click.option("--need_normalization", type=bool, required=False,default=False)
def main(model_path,text,age,gender,emotion,pitch,speed,device,audio_tokenizer_path,output_path,need_normalization):
    webrwkv_model_path = os.path.join(model_path, 'webrwkv.safetensors')
    #1 选择设备
    print(" 💎 选择设备 💎 ")
    adapters = get_available_adapters_py()
    
    if HAS_QUESTIONARY and len(adapters) > 1:
        # 使用交互式菜单选择设备
        choices = [f"{i}: {name}" for i, name in enumerate(adapters)]
        selected = questionary.select(
            "请选择设备 (使用 ↑↓ 键选择，回车确认):",
            choices=choices
        ).ask()
        
        if selected is None:
            print("❌ 未选择设备，程序退出")
            return
            
        adapter_index = int(selected.split(":")[0])
        adapter = adapters[adapter_index]
        print(f"✅ 选择设备: {adapter}")
    else:
        # 回退到传统输入方式
        for index, name in enumerate(adapters):
            print(f'{index}: {name}')
        adapter_index = int(input("请选择设备: "))
        adapter = adapters[adapter_index]
        print(f'选择设备: {adapter}')
    #2 加载模型
    print(" 💎 加载模型 💎 ")
    precision = 'fp32'
    model = Model(webrwkv_model_path, precision, adapter_index)
    print(f'✅ 模型加载成功 {webrwkv_model_path} ✅')
    #3 创建 runtime
    print(" 💎 创建 runtime 💎 ")
    runtime = model.create_thread_runtime()
    print(f'✅ runtime 创建成功 ✅')
    lang = detect_token_lang(text)
    print(f'lang: {lang} before normalization {text}')
    if need_normalization:
        if lang == 'zh':
            text = chinese_normalizer.normalize(text)
        else:
            text = english_normalizer.normalize(text)
    print(f'lang: {lang} after normalization {text}')
    print(f"age: {age}, gender: {gender}, emotion: {emotion}, pitch: {pitch}, speed: {speed}")
    TTS_TAG_0 = 8193
    TTS_TAG_1 = 8194
    TTS_TAG_2 = 8195
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    audio_tokenizer = BiCodecTokenizer(model_dir=audio_tokenizer_path, device=device)

    print(model)
    properties_tokens = convert_standard_properties_to_tokens(age, gender, emotion, pitch, speed)
    print(f'properties_tokens: {properties_tokens}')
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    text_tokens = [i + 8196+4096 for i in text_tokens]
    properties_tokens = tokenizer.encode(properties_tokens, add_special_tokens=False)
    properties_tokens = [i + 8196+4096 for i in properties_tokens]
    print(f'properties_tokens: {properties_tokens}')
    print(f'text_tokens: {text_tokens}')
    all_idx = properties_tokens + [TTS_TAG_2] + text_tokens + [TTS_TAG_0]
    print(f'generate global tokens :{all_idx}')
    import time
    print("💎 Prefill 💎 ")
    start_time = time.time()
    logits = runtime.predict(all_idx)
    end_time = time.time()
    print(f'time: {end_time - start_time}s, prefill speed: {len(all_idx) / (end_time - start_time)} tokens/s')
    print(f"💎 logits 长度{len(logits)} 💎 ")
    print(f"💎 logits 前10个token {logits[:10]} 💎 ")
    print(f"💎 logits 后10个token {logits[-10:]} 💎 ")
    global_tokens_size = 32
    global_tokens = []
    for i in range(global_tokens_size):
        sampled_id = sample_logits(logits[0:4096], temperature=1.0, top_p=0.95, top_k=20)
        global_tokens.append(sampled_id)
        sampled_id += 8196
        logits = runtime.predict_next(sampled_id)
    print(f'generated global_tokens: {global_tokens}')
    global_tokens = [i + 8196 for i in global_tokens]
    print(f'global_tokens: {global_tokens}')
    start_time = time.time()
    x = runtime.predict_next(TTS_TAG_1)
    semantic_tokens = []
    for i in range(2048):
        sampled_id = sample_logits(x[0:8193], temperature=1.0, top_p=0.95, top_k=80)
        if sampled_id == 8192:
            break
        semantic_tokens.append(sampled_id)
        x = runtime.predict_next(sampled_id,)
    print(f'generated semantic_tokens: {semantic_tokens}')
    print(f'semantic_tokens shape: {len(semantic_tokens)}')
    global_tokens = torch.tensor([[i - 8196 for i in global_tokens]], dtype=torch.int32, device=device)
    print(f'global_tokens: {global_tokens}')
    semantic_tokens = torch.tensor([semantic_tokens], dtype=torch.int32, device=device)
    import soundfile as sf
    wav_reconstructed = audio_tokenizer.detokenize(global_tokens, semantic_tokens)
    end_time = time.time()
    print(f'wav_reconstructed shape: {wav_reconstructed.shape}')
    sf.write(output_path, wav_reconstructed, audio_tokenizer.config['sample_rate'])
    print(f'generated wav saved to {output_path}')
    seconds = wav_reconstructed.shape[0] / audio_tokenizer.config['sample_rate']
    print(f'generated wav duration: {seconds} seconds')
    print(f'time: {end_time - start_time}s, RT: {(end_time - start_time)/seconds} ')
if __name__ == "__main__":
    main()