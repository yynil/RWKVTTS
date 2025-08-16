import click
from webrwkv_py import Model, ThreadRuntime, get_available_adapters_py
from transformers import AutoTokenizer
import torch
import numpy as np
from scipy import special
from scipy.stats import rv_discrete
from torch.nn import functional as F
from math import inf
from utils.utilities import generate_global_tokens,generate_input_embeddings,extract_embeddings_for_global_tokens,convert_standard_properties_to_tokens
import re
import os
import onnxruntime as ort
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
    
    # 将 List[float] 转换为 numpy 数组
    if isinstance(logits, list):
        logits = np.array(logits)
    
    # 使用 scipy 的 softmax
    probs = special.softmax(logits, axis=-1)
    top_k = int(top_k)
    

    
    # 使用 numpy 进行采样
    sorted_ids = np.argsort(probs)
    sorted_probs = probs[sorted_ids][::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    
    # 找到 top_p 的截止点
    cutoff_mask = cumulative_probs >= top_p
    if np.any(cutoff_mask):
        cutoff_idx = np.argmax(cutoff_mask)
        cutoff = float(sorted_probs[cutoff_idx])
        probs[probs < cutoff] = 0
    
    # 应用 top_k 过滤
    if top_k < len(probs) and top_k > 0:
        probs[sorted_ids[:-top_k]] = 0
    
    # 应用温度
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)
    
    # 重新归一化概率
    probs = probs / np.sum(probs)
    
    # 使用 numpy 的随机选择
    out = np.random.choice(a=len(probs), size=1, p=probs)
    return int(out[0])



@click.command()
@click.option("--model_path", type=str, required=True)
@click.option("--text", type=str, required=True)
@click.option("--age", type=str, required=False,default='middle-aged')
@click.option("--gender", type=str, required=False,default='female')
@click.option("--emotion", type=str, required=False,default='neutral')
@click.option("--pitch", type=str, required=False,default='medium_pitch')
@click.option("--speed", type=str, required=False,default='medium')
@click.option("--device", type=str, required=False,default='cpu')
@click.option("--decoder_path", type=str, required=False,default='/Volumes/bigdata/models/BiCodecDetokenize.onnx')
@click.option("--output_path", type=str, required=False,default="generated_from_webrwkvosx.wav")
@click.option("--need_normalization", type=bool, required=False,default=False)
def main(model_path,text,age,gender,emotion,pitch,speed,device,decoder_path,output_path,need_normalization):
    webrwkv_model_path = os.path.join(model_path, 'webrwkv.safetensors')
    #1 选择设备
    print(" 💎 选择设备 💎 ")
    adapters = get_available_adapters_py()
    
    if HAS_QUESTIONARY and len(adapters) >= 1:
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
    print(f"age: {age}, gender: {gender}, emotion: {emotion}, pitch: {pitch}, speed: {speed}")
    TTS_TAG_0 = 8193
    TTS_TAG_1 = 8194
    TTS_TAG_2 = 8195
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)

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
    global_tokens_size = 32
    global_tokens = []
    start_time = time.time()
    for i in range(global_tokens_size):
        sampled_id = sample_logits(logits[0:4096], temperature=1.0, top_p=0.95, top_k=20)
        global_tokens.append(sampled_id)
        sampled_id += 8196
        logits = runtime.predict_next(sampled_id)
    end_time = time.time()
    print(f'time: {end_time - start_time}s, global_tokens speed: {32 / (end_time - start_time)} tokens/s')
    global_tokens = [i + 8196 for i in global_tokens]
    start_time = time.time()
    x = runtime.predict_next(TTS_TAG_1)
    semantic_tokens = []
    start_time = time.time()
    for i in range(2048):
        sampled_id = sample_logits(x[0:8193], temperature=1.0, top_p=0.95, top_k=80)
        if sampled_id == 8192:
            break
        semantic_tokens.append(sampled_id)
        x = runtime.predict_next(sampled_id,)
    end_time = time.time()
    print(f'time: {end_time - start_time}s, semantic_tokens speed: {len(semantic_tokens) / (end_time - start_time)} tokens/s')
    global_tokens = torch.tensor([[i - 8196 for i in global_tokens]], dtype=torch.int32, device=device)
    semantic_tokens = torch.tensor([semantic_tokens], dtype=torch.int32, device=device)
    import soundfile as sf
    print(f'🎿Start to load onnx model')
    ort_session = ort.InferenceSession(decoder_path)
    print(f'⛷️Load onnx model success')
    global_tokens = np.array(global_tokens, dtype=np.int64).reshape(1,1,-1)
    semantic_tokens = np.array(semantic_tokens, dtype=np.int64).reshape(1,-1)
    start_time = time.time()
    outputs = ort_session.run(None, {"global_tokens": global_tokens, "semantic_tokens": semantic_tokens})
    wav_reconstructed = outputs[0].reshape(-1)
    end_time = time.time()
    print(f'time to decode wav: {end_time - start_time}s, speed: {len(wav_reconstructed) / (end_time - start_time)} tokens/s')
    sf.write(output_path, wav_reconstructed, 16000)
    print(f'generated wav saved to {output_path}')
if __name__ == "__main__":
    main()