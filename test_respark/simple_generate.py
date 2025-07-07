import torch
import numpy as np
import soundfile as sf
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from utils.utilities import generate_embeddings

def simple_generate_speech(model, tokenizer, text, bicodec, prompt_audio, 
                          max_new_tokens=1000, temperature=0.8, device="cuda:0"):
    """
    简化的语音生成函数
    
    Args:
        model: 语言模型
        tokenizer: 文本分词器
        text: 要生成语音的文本
        bicodec: BiCodecTokenizer 实例
        prompt_audio: 提示音频数组
        max_new_tokens: 最大生成token数
        temperature: 温度参数
        device: 设备
    
    Returns:
        wav: 生成的音频波形
    """
    # 设置eos_token_id
    eos_token_id = model.config.vocab_size - 1
    print(f"EOS token ID: {eos_token_id}")
    
    # 生成输入嵌入
    embeddings = generate_embeddings(
        model=model,
        tokenizer=tokenizer,
        text=text,
        bicodec=bicodec,
        prompt_text=None,
        prompt_audio=prompt_audio
    )
    
    print(f"输入嵌入形状: {embeddings['input_embs'].shape}")
    
    # 生成
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=embeddings['input_embs'],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    # 提取新生成的token
    input_length = embeddings['input_embs'].shape[1]
    generated_tokens = outputs[0][input_length:]
    
    print(f"生成的token数量: {len(generated_tokens)}")
    print(f"生成的token IDs: {generated_tokens.tolist()}")
    
    # 直接使用生成的token ID作为semantic tokens
    semantic_tokens_tensor = generated_tokens.unsqueeze(0).to(device)
    
    # 提取global tokens
    global_tokens, _ = bicodec.tokenize(prompt_audio)
    global_tokens = global_tokens.squeeze(0).squeeze(0)
    
    # 解码音频
    with torch.no_grad():
        wav = bicodec.detokenize(global_tokens.unsqueeze(0), semantic_tokens_tensor)
    
    return wav

def main():
    # 配置参数
    model_dir = "/home/yueyulin/models/rwkv7-0.4B-g1-respark/"
    spark_model_dir = "/home/yueyulin/models/Spark-TTS-0.5B/"
    prompt_audio_path = "/path/to/your/prompt_audio.wav"  # 请替换为实际的音频路径
    output_path = "generated_speech.wav"
    text = "你好，这是一个测试。"
    device = "cuda:0"
    
    # 加载模型和分词器
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    bicodec = BiCodecTokenizer(spark_model_dir, device=device)
    
    model.to(device)
    model.eval()
    
    # 加载提示音频
    print("加载提示音频...")
    prompt_audio, sr = sf.read(prompt_audio_path)
    if sr != 16000:
        from librosa import resample
        prompt_audio = resample(prompt_audio, orig_sr=sr, target_sr=16000)
        prompt_audio = np.array(prompt_audio, dtype=np.float32)
    
    # 生成语音
    print("开始生成语音...")
    wav = simple_generate_speech(
        model=model,
        tokenizer=tokenizer,
        text=text,
        bicodec=bicodec,
        prompt_audio=prompt_audio,
        max_new_tokens=1000,
        temperature=0.8,
        device=device
    )
    
    if wav is not None:
        sf.write(output_path, wav, 16000)
        print(f"语音已保存到: {output_path}")
    else:
        print("生成失败")

if __name__ == "__main__":
    main() 