import click
import os
os.environ["RWKV_V7_ON"] = "1" # enable this for rwkv-7 models
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1'
from rwkv.model import RWKV
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from transformers import AutoTokenizer
import torch
import numpy as np
from torch.nn import functional as F
from math import inf
from utils.utilities import generate_global_tokens,generate_input_embeddings,extract_embeddings_for_global_tokens,convert_standard_properties_to_tokens
def sample_logits(logits, temperature=1.0, top_p=0.85, top_k=0,black_list_tokens=[]):
    if temperature == 0:
        temperature = 1.0
        top_p = 0
    probs = F.softmax(logits.float(), dim=-1)
    top_k = int(top_k)
    if black_list_tokens is not None:
        probs[black_list_tokens] = -inf
    # 'privateuseone' is the type of custom devices like `torch_directml.device()`
    if probs.device.type in ['cpu', 'privateuseone']:
        probs = probs.cpu().numpy()
        sorted_ids = np.argsort(probs)
        sorted_probs = probs[sorted_ids][::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
        probs[probs < cutoff] = 0
        if top_k < len(probs) and top_k > 0:
            probs[sorted_ids[:-top_k]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        probs = probs / np.sum(probs)
        out = np.random.choice(a=len(probs), p=probs)
        return int(out)
    else:
        sorted_ids = torch.argsort(probs)
        sorted_probs = probs[sorted_ids]
        sorted_probs = torch.flip(sorted_probs, dims=(0,))
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
        probs[probs < cutoff] = 0
        if top_k < len(probs) and top_k > 0:
            probs[sorted_ids[:-top_k]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        out = torch.multinomial(probs, num_samples=1)[0]
        return int(out)



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
@click.option("--output_path", type=str, required=False,default="generated_from_chatrwkv.wav")
def main(model_path,text,age,gender,emotion,pitch,speed,device,audio_tokenizer_path,output_path):
    print(f"age: {age}, gender: {gender}, emotion: {emotion}, pitch: {pitch}, speed: {speed}")
    TTS_TAG_0 = 8193
    TTS_TAG_1 = 8194
    TTS_TAG_2 = 8195
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    audio_tokenizer = BiCodecTokenizer(model_dir=audio_tokenizer_path, device=device)
    black_list_tokens = [id for id in range(4096,8193)]
    model = RWKV(model=f"{model_path}/model_converted",  strategy='cuda:0 bf16')
    print(model)
    properties_tokens = convert_standard_properties_to_tokens(age, gender, emotion, pitch, speed)
    print(f'properties_tokens: {properties_tokens}')
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    text_tokens = [i + 8196+4096 for i in text_tokens]
    properties_tokens = tokenizer.encode(properties_tokens, add_special_tokens=False)
    properties_tokens = [i + 8196+4096 for i in properties_tokens]
    print(f'properties_tokens: {properties_tokens}')
    print(f'text_tokens: {text_tokens}')
    print(model.z['emb.weight'].shape)
    print(model.z['head.weight'].shape)
    all_idx = properties_tokens + [TTS_TAG_2] + text_tokens + [TTS_TAG_0]
    print(f'generate global tokens :{all_idx}')
    import time
    start_time = time.time()
    x,state = model.forward(all_idx, None)
    end_time = time.time()
    print(f'time: {end_time - start_time}s, prefill speed: {len(all_idx) / (end_time - start_time)} tokens/s')
    print(f'x: {x.shape}')
    global_tokens_size = 32
    global_tokens = []
    for i in range(global_tokens_size):
        sampled_id = sample_logits(x, temperature=1.0, top_p=0.95, top_k=20,black_list_tokens=black_list_tokens)
        global_tokens.append(sampled_id)
        sampled_id += 8196
        x,state = model.forward([sampled_id], state)
    print(f'generated global_tokens: {global_tokens}')
    global_tokens = [i + 8196 for i in global_tokens]
    print(f'global_tokens: {global_tokens}')
    x,state = model.forward([TTS_TAG_1], state)
    semantic_tokens = []
    for i in range(2048):
        sampled_id = sample_logits(x, temperature=1.0, top_p=0.95, top_k=50)
        if sampled_id == 8192:
            break
        semantic_tokens.append(sampled_id)
        x,state = model.forward([sampled_id], state)
    print(f'generated semantic_tokens: {semantic_tokens}')
    print(f'semantic_tokens shape: {len(semantic_tokens)}')
    global_tokens = torch.tensor([[i - 8196 for i in global_tokens]], dtype=torch.int32, device=device)
    print(f'global_tokens: {global_tokens}')
    semantic_tokens = torch.tensor([semantic_tokens], dtype=torch.int32, device=device)
    import soundfile as sf
    wav_reconstructed = audio_tokenizer.detokenize(global_tokens, semantic_tokens)
    print(f'wav_reconstructed shape: {wav_reconstructed.shape}')
    sf.write(output_path, wav_reconstructed, audio_tokenizer.config['sample_rate'])
    print(f'generated wav saved to {output_path}')
if __name__ == "__main__":
    main()