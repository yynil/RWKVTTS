import click
from webrwkv_py import Model, ThreadRuntime, get_available_adapters_py
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from transformers import AutoTokenizer
import torch
from torch.nn import functional as F
from math import inf
import re
import os
import numpy as np
import onnxruntime as ort
from pathlib import Path
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
    """从logits中采样token"""
    if temperature == 0:
        temperature = 1.0
        top_p = 0
    
    if isinstance(logits, list):
        logits = np.array(logits)
    
    try:
        from scipy import special
        probs = special.softmax(logits, axis=-1)
    except ImportError:
        # 如果没有scipy，使用numpy的简单实现
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
    
    top_k = int(top_k)
    
    sorted_ids = np.argsort(probs)
    sorted_probs = probs[sorted_ids][::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    
    cutoff_mask = cumulative_probs >= top_p
    if np.any(cutoff_mask):
        cutoff_idx = np.argmax(cutoff_mask)
        cutoff = float(sorted_probs[cutoff_idx])
        probs[probs < cutoff] = 0
    
    if top_k < len(probs) and top_k > 0:
        probs[sorted_ids[:-top_k]] = 0
    
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)
    
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), size=1, p=probs)
    return int(out[0])



@click.command()
@click.option("--model_path", type=str, required=True)
@click.option("--text", type=str, required=True)
@click.option("--prompt_audio",type=str,required=False,default="zero_shot_prompt.wav")
@click.option("--prompt_text",type=str,required=False,default="希望你以后能够做的，比我还好呦！")
@click.option("--device", type=str, required=False,default='cpu')
@click.option("--output_path", type=str, required=False,default="generated_from_webrwkv_zeroshot.wav")
@click.option("--using_prompt_text",type=bool,required=False,default=False)
@click.option("--spark_bicodec_path",type=str,required=False,default=None)
def main(model_path,text,prompt_audio,prompt_text,device,output_path,using_prompt_text,spark_bicodec_path):
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
    TTS_TAG_0 = 8193
    TTS_TAG_1 = 8194
    TTS_TAG_2 = 8195
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    print(model)
    text_tokens = tokenizer.encode(prompt_text+text if using_prompt_text else text, add_special_tokens=False)
    text_tokens = [i + 8196+4096 for i in text_tokens]
    print(f'text_tokens: {text_tokens}')
    import time
    if spark_bicodec_path is None:
        from utils.ref_audio_utilities_improved import RefAudioUtilitiesImproved 
        audio_tokenizer_path = os.path.join(model_path, 'BiCodecTokenize.onnx')
        wav2vec2_path = os.path.join(model_path, 'wav2vec2-large-xlsr-53.onnx')
        ref_audio_utilities = RefAudioUtilitiesImproved(audio_tokenizer_path, wav2vec2_path)
        global_tokens, prompt_semantic_tokens = ref_audio_utilities.tokenize(Path(prompt_audio))
        print(f'global_tokens: {global_tokens}')
        print(f'semantic_tokens: {prompt_semantic_tokens}')
    else:
        from sparktts.models.audio_tokenizer import BiCodecTokenizer
        print(f'🔍 使用Spark-TTS-0.5B的BiCodecTokenizer')
        audio_tokenizer = BiCodecTokenizer(model_dir=spark_bicodec_path, device=device)
        global_tokens, prompt_semantic_tokens = audio_tokenizer.tokenize(Path(prompt_audio))
        print(f'global_tokens: {global_tokens}')
        print(f'semantic_tokens: {prompt_semantic_tokens}')
    
    # 直接使用flatten()展平数组并转换为Python一维数组
    global_tokens = [int(i) + 8196 for i in global_tokens.flatten()]
    prompt_semantic_tokens = [int(i) for i in prompt_semantic_tokens.flatten()]

    all_idx =[TTS_TAG_2] + text_tokens + [TTS_TAG_0] + global_tokens + [TTS_TAG_1] 
    if using_prompt_text:
        all_idx = all_idx + prompt_semantic_tokens
    start_time = time.time()
    x = runtime.predict(all_idx)
    semantic_tokens = []
    for i in range(2048):
        sampled_id = sample_logits(x[0:8193], temperature=1.0, top_p=0.95, top_k=80)
        if sampled_id == 8192:
            break
        semantic_tokens.append(sampled_id)
        x = runtime.predict_next(sampled_id,)
    print(f'generated semantic_tokens: {semantic_tokens}')
    print(f'semantic_tokens shape: {len(semantic_tokens)}')
    global_tokens = [i - 8196 for i in global_tokens]
    print(f'global_tokens: {global_tokens}')
    if spark_bicodec_path is None:
        decoder_path = os.path.join(model_path, "BiCodecDetokenize.onnx")
        print(f"🔍 自动设置解码器路径: {decoder_path}")
        ort_session = ort.InferenceSession(decoder_path)
        global_tokens = np.array(global_tokens, dtype=np.int64).reshape(1, 1, -1)
        semantic_tokens = np.array(semantic_tokens, dtype=np.int64).reshape(1, -1)
        outputs = ort_session.run(None, {
                    "global_tokens": global_tokens, 
                    "semantic_tokens": semantic_tokens
                })
        wav_data = outputs[0].reshape(-1)
        import soundfile as sf
        end_time = time.time()
        print(f'wav_reconstructed shape: {wav_data.shape}')
        sf.write(output_path, wav_data, 16000)
        print(f'generated wav saved to {output_path}')
    else:
        global_tokens = torch.tensor([[i for i in global_tokens]], dtype=torch.int32, device=device)
        print(f'global_tokens: {global_tokens}')
        semantic_tokens = torch.tensor([semantic_tokens], dtype=torch.int32, device=device)
        from sparktts.models.audio_tokenizer import BiCodecTokenizer
        print(f'🔍 使用Spark-TTS-0.5B的BiCodecTokenizer')
        wav_data = audio_tokenizer.detokenize(global_tokens, semantic_tokens)
        import soundfile as sf
        end_time = time.time()
        print(f'wav_reconstructed shape: {wav_data.shape}')
        sf.write(output_path, wav_data, 16000)
        print(f'generated wav saved to {output_path}')
if __name__ == "__main__":
    main()