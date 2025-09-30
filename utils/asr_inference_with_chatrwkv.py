import os
os.environ['RWKV_CUDA_ON'] = '1'
os.environ['RWKV_V7_ON'] = '1'
import torch
from dataclasses import dataclass
from model.llm.rwkv_asr_whisper import load_whisper_feature_extractor_and_encoder
from rwkv.model import RWKV, RWKV_x070_CMix_seq, RWKV_x070_TMix_seq, RWKV_x070_CMix_one, RWKV_x070_TMix_one
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
import soundfile as sf
import torch.nn.functional as F
from typing import List
from transformers import WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder
import numpy as np
import click
@dataclass
class AsrModels:
    audio_llm: RWKV
    whisper_feature_extractor: WhisperFeatureExtractor
    whisper_encoder: WhisperEncoder
    project1_linear: torch.nn.Linear
    project2_linear: torch.nn.Linear
    llm: RWKV
    tokenizer: TRIE_TOKENIZER


def forward_one_with_embeds(model :RWKV,embeds:torch.Tensor,state:List[torch.Tensor]):
    with torch.no_grad(): 
        z = model.z
        x = embeds

        v_first = torch.empty_like(x)
        for i in range(model.n_layer):
            bbb = f'blocks.{i}.'
            att = f'blocks.{i}.att.'
            ffn = f'blocks.{i}.ffn.'

            xx = F.layer_norm(x, (model.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

            xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_one(i, model.n_head, model.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                z[att+'ln_x.weight'], z[att+'ln_x.bias'])
            x = x + xx

            xx = F.layer_norm(x, (model.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

            xx, state[i*3+2] = RWKV_x070_CMix_one(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
            x = x + xx
        
            # if math.isnan(torch.min(x).item()): print(idx, i)

        x = F.layer_norm(x, (model.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
        x = x @ z['head.weight']
        return x, state

def forward_seq_with_embeds(model :RWKV,embeds:torch.Tensor,dtype,device,state:List[torch.Tensor] = None,return_whole_output:bool = True):
    if state == None:
        state = [None for _ in range(model.n_layer * 3)]
        for i in range(model.n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
            state[i*3+0] = torch.zeros(model.n_embd, dtype=dtype, requires_grad=False, device=device)
            state[i*3+1] = torch.zeros((model.n_embd // model.head_size, model.head_size, model.head_size), dtype=torch.float, requires_grad=False, device=device)
            state[i*3+2] = torch.zeros(model.n_embd, dtype=dtype, requires_grad=False, device=device)
    z = model.z
    with torch.no_grad(): 
        x = embeds
        v_first = torch.empty_like(x)
        for i in range(model.n_layer):
            bbb = f'blocks.{i}.'
            att = f'blocks.{i}.att.'
            ffn = f'blocks.{i}.ffn.'

            xx = F.layer_norm(x, (model.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

            xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_seq(i, model.n_head, model.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                        z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                        z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                        z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                        z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                        z[att+'ln_x.weight'], z[att+'ln_x.bias'])
            x = x + xx

            xx = F.layer_norm(x, (model.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

            xx, state[i*3+2] = RWKV_x070_CMix_seq(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
            x = x + xx
        if not return_whole_output:
            x = x[-1,:]
        x = F.layer_norm(x, (model.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
        return x, state


def load_asr_models(audio_lm_path, llm_path,whisper_path,tokenizer_path,device,dtype):
    whisper_feature_extractor, whisper_encoder = load_whisper_feature_extractor_and_encoder(whisper_path)
    audil_lm_model_name = audio_lm_path + "/model_converted"
    audio_llm = RWKV(model=f"{audil_lm_model_name}",  strategy=f'{device} { "bf16" if dtype==torch.bfloat16 else "fp16"}')
    project1 = torch.load(audio_lm_path + "/projector1.pt")
    project2 = torch.load(audio_lm_path + "/projector2.pt")
    llm_model_name = llm_path.replace('.pth', '')
    llm = RWKV(model=f"{llm_model_name}",  strategy=f'{device} { "bf16" if dtype==torch.bfloat16 else "fp16"}')
    project1_linear = torch.nn.Linear(project1['weight'].shape[1], project1['weight'].shape[0])
    project1_linear.load_state_dict(project1)
    project2_linear = torch.nn.Linear(project2['weight'].shape[1], project2['weight'].shape[0])
    project2_linear.load_state_dict(project2)
    tokenizer = TRIE_TOKENIZER(tokenizer_path)
    return AsrModels(
        audio_llm=audio_llm,
        whisper_feature_extractor=whisper_feature_extractor,
        whisper_encoder=whisper_encoder.to(device),
        project1_linear=project1_linear.to(device=device,dtype=dtype),
        project2_linear=project2_linear.to(device=device,dtype=dtype),
        llm=llm,
        tokenizer=tokenizer,
    )

def sample_logits(logits, temperature=1.0, top_p=0.85, top_k=0):
    if temperature == 0:
        temperature = 1.0
        top_p = 0
    probs = F.softmax(logits.float(), dim=-1)
    top_k = int(top_k)
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

def extract_audio_latents(models, audio_file_path,dtype):
    audio_data,sample_rate = sf.read(audio_file_path)
    if sample_rate != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
    audio_data = audio_data.reshape(1,-1)
    print(f'type of audio_data: {type(audio_data)}')
    print(f'shape of audio_data: {audio_data.shape}')
    print(f'sample_rate: {sample_rate}')
    if len(audio_data.shape) == 2:
        audio_data = audio_data.squeeze(0)
    features = models.whisper_feature_extractor([audio_data], sampling_rate=16000, return_tensors="pt", return_attention_mask=True, padding_value=0.0)
    audio_attention_mask = features['attention_mask']
    feature_dtype = next(models.whisper_encoder.parameters()).dtype
    device = next(models.whisper_encoder.parameters()).device
    input_features = features['input_features'].to(dtype=feature_dtype).to(device)
    audio_attention_mask = audio_attention_mask.to(device)
    with torch.no_grad():
        encoder_outputs = models.whisper_encoder(input_features, attention_mask=audio_attention_mask)
    audio_latents = encoder_outputs.last_hidden_state  # [1, T_audio, hidden_size]

    if audio_attention_mask.shape[1] != audio_latents.shape[1]:
        # 计算下采样比例
        downsample_ratio = audio_attention_mask.shape[1] / audio_latents.shape[1]
    else:
        downsample_ratio = 1.0
        
    # 获取有效音频长度（考虑下采样）
    audio_valid_length = int(audio_attention_mask.sum().item() / downsample_ratio)+1
    audio_latents = audio_latents.to(dtype=dtype)
    print(f'type of audio_latents: {type(audio_latents)}')
    print(f'shape of audio_latents: {audio_latents.shape}')
    print(f'dtype of audio_latents: {audio_latents.dtype}')
    projected_latents = models.project1_linear(audio_latents)
    return projected_latents,audio_valid_length

@torch.inference_mode()
def inference_asr(models, audio_path, language,dtype,device):
    if language == 'chinese':
        print(f'language: {language}')
        instruction = "User: 请将以下语音转写为中文。\n"
        hints = "Assistant: "
    else:
        print(f'language: {language}')
        instruction = "User: Convert the audios to English.\n"
        hints = "Assistant: "
    
    print(f'load audio from {audio_path}')
    audio_path = audio_path
    audio_latents,audio_valid_length = extract_audio_latents(models, audio_path,dtype)
    audio_latents = audio_latents.squeeze(0)
    with torch.no_grad():
        audio_latents = F.layer_norm(audio_latents, (models.audio_llm.n_embd,), weight=models.audio_llm.z['blocks.0.ln0.weight'], bias=models.audio_llm.z['blocks.0.ln0.bias'])#do the first layer norm for embeddings input
    audio_latents, _ = forward_seq_with_embeds(models.audio_llm, audio_latents, dtype, device, None, True)
    audio_latents = models.project2_linear(audio_latents)
    audio_latents = audio_latents[:audio_valid_length]
    instruction_input_ids = models.tokenizer.encode(instruction)
    hints_input_ids = models.tokenizer.encode(hints)
    instruction_input_embeds = models.llm.z['emb.weight'][instruction_input_ids]#first layer norm is done when the embeddings are loaded
    hints_input_embeds = models.llm.z['emb.weight'][hints_input_ids]#first layer norm is done when the embeddings are loaded
    with torch.no_grad():
        audio_latents = F.layer_norm(audio_latents, (models.llm.n_embd,), weight=models.llm.z['blocks.0.ln0.weight'], bias=models.llm.z['blocks.0.ln0.bias'])#do the first layer norm for embeddings input
    whole_input_embeds = torch.cat([instruction_input_embeds, audio_latents, hints_input_embeds], dim=0)
    hidden_states,state = forward_seq_with_embeds(models.llm, whole_input_embeds, dtype, device, None, False)
    with torch.no_grad():
        logits = hidden_states @ models.llm.z['head.weight']
    next_token = sample_logits(logits,top_k=10,top_p=0.95,temperature=1)
    results = []
    results.append(next_token)
    while len(results) < 1024:
        logits,state = models.llm.forward([next_token], state)
        next_token = sample_logits(logits,top_k=10,top_p=0.95,temperature=1)
        if next_token == 0:
            break
        results.append(next_token)
    return results

@click.command()
@click.option('--audio-lm-path', default="/home/yueyulin/models/rwkv7_0.1b_audio_lm_latents_1.5b_44k", 
              help='音频语言模型路径')
@click.option('--llm-path', default="/home/yueyulin/models/rwkv7-g1a-1.5b-20250922-ctx4096.pth", 
              help='大语言模型路径')
@click.option('--whisper-path', default="/home/yueyulin/models/whisper-large-v3/", 
              help='Whisper模型路径')
@click.option('--audio-path', default="/home/yueyulin/github/RWKVTTS/my_chinese.wav", 
              help='音频文件路径')
@click.option('--tokenizer-path', default="tokenizer/rwkv_vocab_v20230424.txt", 
              help='分词器路径')
@click.option('--language', default="chinese", 
              help='语言类型 (chinese/english)')
@click.option('--device', default="cuda:0", 
              help='设备类型 (cuda:0/cpu)')
@click.option('--dtype', default="float16", 
              type=click.Choice(['float16', 'float32', 'bfloat16']),
              help='数据类型')
def main(audio_lm_path, llm_path, whisper_path, audio_path, tokenizer_path, language, device, dtype):
    """
    主函数，用于运行ASR推理
    """
    # 转换dtype字符串为torch类型
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16
    }
    dtype = dtype_map[dtype]
    models = load_asr_models(audio_lm_path, llm_path, whisper_path, tokenizer_path, device, dtype)
    print(f'audio_llm: {models.audio_llm}')
    print(f'whisper_feature_extractor: {models.whisper_feature_extractor}')
    print(f'whisper_encoder: {models.whisper_encoder}')
    print(f'llm: {models.llm}')
    print(f'project1: {models.project1_linear}')
    print(f'project2: {models.project2_linear}')
    results = inference_asr(models, audio_path, language, dtype, device)
    print(f'results: {results}')
    print(f'decode results: {models.tokenizer.decode(results)}')
    return results

if __name__ == "__main__":
    main()