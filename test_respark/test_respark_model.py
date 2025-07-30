import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from utils.utilities import generate_embeddings
import re
import soundfile as sf

def generate_speech(model, tokenizer, text, bicodec, prompt_text=None, prompt_audio=None, 
                   max_new_tokens=3000, do_sample=True, top_k=50, top_p=0.95, 
                   temperature=1.0, device="cuda:0", eos_token_id=8192):
    """
    生成语音的函数
    
    Args:
        model: 语言模型
        tokenizer: 文本分词器
        text: 要生成语音的文本
        bicodec: BiCodecTokenizer 实例
        prompt_text: 提示文本（可选）
        prompt_audio: 提示音频数组（可选）
        max_new_tokens: 最大生成token数
        do_sample: 是否使用采样
        top_k: top-k采样参数
        top_p: top-p采样参数
        temperature: 温度参数
        device: 设备
    
    Returns:
        wav: 生成的音频波形
    """
    print(f"EOS token ID: {eos_token_id}")
    
    # 生成输入嵌入
    embeddings = generate_embeddings(
        model=model,
        tokenizer=tokenizer,
        text=text,
        bicodec=bicodec,
        prompt_text=prompt_text,
        prompt_audio=prompt_audio
    )
    
    print("开始生成语音...")
    print(f"输入嵌入形状: {embeddings['input_embs'].shape}")
    global_tokens = embeddings['global_tokens'].unsqueeze(0)
    # 设置模型为评估模式
    model.eval()
    
    with torch.no_grad():
        # 使用模型的generate方法
        generated_outputs = model.generate(
            inputs_embeds=embeddings['input_embs'],
            attention_mask=torch.ones((1, embeddings['input_embs'].shape[1]),dtype=torch.long,device=device),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else tokenizer.eos_token_id,
            use_cache=True
        )
    print(f"generated_outputs: {generated_outputs}")
    
    print(f"生成的token数量: {generated_outputs.shape}")
    print(f"生成的token IDs: {generated_outputs.tolist()}")
    
    # 直接使用生成的token ID作为semantic tokens
    # 注意：这里生成的token ID是模型词表中的ID，不是原始tokenizer的词表
    semantic_tokens_tensor = generated_outputs[:,:-1]
    
    print(f"Semantic tokens shape: {semantic_tokens_tensor.shape}")
    
    
    print(f"Global tokens shape: {global_tokens.shape}")
    
    # 使用BiCodec解码生成音频
    with torch.no_grad():
        wav = bicodec.detokenize(global_tokens, semantic_tokens_tensor)
    
    print(f"生成的音频形状: {wav.shape}")
    return wav

def save_audio(wav, output_path, sample_rate=16000):
    """
    保存音频文件
    
    Args:
        wav: 音频波形数组
        output_path: 输出文件路径
        sample_rate: 采样率
    """
    sf.write(output_path, wav, sample_rate)
    print(f"音频已保存到: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/home/yueyulin/models/rwkv7-0.4B-g1-respark/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ckpt_file", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--text", type=str, default="Hello, how are you?")
    parser.add_argument("--spark_model_dir", type=str, default="/home/yueyulin/models/Spark-TTS-0.5B/")
    parser.add_argument("--prompt_text", type=str, default=None)
    parser.add_argument("--prompt_audio_path", type=str, required=True, help="提示音频文件路径（必需）")
    parser.add_argument("--output_path", type=str, default="generated_speech.wav", help="输出音频文件路径")
    parser.add_argument("--max_new_tokens", type=int, default=3000, help="最大生成token数")
    parser.add_argument("--do_sample", type=bool, default=True, help="是否使用采样")
    parser.add_argument("--top_k", type=int, default=50, help="top-k采样参数")
    parser.add_argument("--top_p", type=float, default=0.95, help="top-p采样参数")
    parser.add_argument("--temperature", type=float, default=1.0, help="温度参数")
    parser.add_argument("--eos_token_id", type=int, default=8192, help="EOS token ID")
    args = parser.parse_args()

    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError(f"Invalid dtype: {args.dtype}")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=dtype, trust_remote_code=True)
    print("Model loaded successfully")
    print(f"Model type: {type(model)}")
    if args.ckpt_file:
        model.load_state_dict(torch.load(args.ckpt_file, map_location=args.device))
        print(f"Loaded checkpoint from {args.ckpt_file}")
    else:
        print("No checkpoint file provided using the model weights")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    print("Tokenizer loaded successfully")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # 加载 BiCodecTokenizer
    bicodec = BiCodecTokenizer(args.spark_model_dir, device=args.device)
    print("BiCodecTokenizer loaded successfully")

    # 将模型移动到指定设备
    model.to(args.device)
    model.eval()

    # 处理提示音频
    prompt_audio = None
    if args.prompt_audio_path:
        prompt_audio, sampling_rate = sf.read(args.prompt_audio_path)
        print(f"Loaded prompt audio from {args.prompt_audio_path}")
        print(f"Original sampling rate: {sampling_rate}Hz")
        print(f"Audio shape: {prompt_audio.shape}")
        
        # 检查并处理采样率
        target_sample_rate = bicodec.config['sample_rate']
        if sampling_rate != target_sample_rate:
            print(f"Resampling from {sampling_rate}Hz to {target_sample_rate}Hz...")
            from librosa import resample
            prompt_audio = resample(prompt_audio, orig_sr=sampling_rate, target_sr=target_sample_rate)
            prompt_audio = np.array(prompt_audio, dtype=np.float32)
            print(f"Resampled audio shape: {prompt_audio.shape}")
        else:
            print(f"Audio sampling rate already matches target ({target_sample_rate}Hz)")
    else:
        raise ValueError("prompt_audio_path 是必需的！generate_embeddings 函数需要提示音频来提取 global_tokens 和 semantic_tokens。请使用 --prompt_audio_path 参数指定音频文件路径。")

    # 生成语音
    print(f"\n开始生成语音...")
    print(f"输入文本: '{args.text}'")
    print(f"提示文本: {args.prompt_text}")
    print(f"提示音频: {'Yes' if prompt_audio is not None else 'No'}")
    
    try:
        wav = generate_speech(
            model=model,
            tokenizer=tokenizer,
            text=args.text,
            bicodec=bicodec,
            prompt_text=args.prompt_text,
            prompt_audio=prompt_audio,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            device=args.device,
            eos_token_id=args.eos_token_id
        )
        
        if wav is not None:
            # 保存生成的音频
            save_audio(wav, args.output_path, sample_rate=bicodec.config['sample_rate'])
            print("语音生成完成！")
        else:
            print("语音生成失败！")
        
    except Exception as e:
        print(f"生成语音时出错: {e}")
        import traceback
        traceback.print_exc()
