import os
import argparse
from argparse import Namespace
import torch
from model.llm.rwkv_asr_cuda_whisper import RWKV7ASRModelCuda, load_whisper_feature_extractor_and_encoder, RWKV7ModelForLatentInputsCuda, RWKV7ModelForCausalLMCuda
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
import soundfile as sf
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=False, default="/home/yueyulin/rwkv7_whisper_cuda")
    parser.add_argument("--ckpt_file", type=str, required=False, default="/home/yueyulin/rwkv7_whisper_cuda_training/epoch_0_step_100000/pytorch_model.bin")
    parser.add_argument("--device", type=str, required=False, default="cuda:0")
    parser.add_argument("--audio_path", type=str, required=False, default="/home/yueyulin/github/RWKVTTS/my_chinese.wav")
    parser.add_argument("--language", type=str, required=False, default="chinese")
    args = parser.parse_args()
    import json
    device = args.device
    model_dir = args.model_dir
    audio_path = args.audio_path
    language = args.language
    if language == 'chinese':
        print(f'language: {language}')
        instruction = "User: 请将以下语音转写为中文。\n"
        hints = "Assistant: "
    else:
        print(f'language: {language}')
        instruction = "User: Convert the audios to English.\n"
        hints = "Assistant: "
    whisper_encoder_path = os.path.join(model_dir, "whisper_encoder")
    whisper_feature_extractor, whisper_encoder = load_whisper_feature_extractor_and_encoder(whisper_encoder_path)
    print(whisper_feature_extractor)
    print(whisper_encoder)
    llm_args_path = os.path.join(model_dir, "llm_args.json")
    with open(llm_args_path, "r") as f:
        llm_args = json.load(f)
    llm_args = Namespace(**llm_args)
    llm_args.grad_cp = 0
    llm_args.dropout = 0
    print(llm_args) 
    llm = RWKV7ModelForCausalLMCuda(llm_args)
    print(llm)
    llm = llm.to(device)
    print(llm)
    audio_lm_args_path = os.path.join(model_dir, "audio_lm_args.json")
    with open(audio_lm_args_path, "r") as f:
        audio_lm_args = json.load(f)
    audio_lm_args = Namespace(**audio_lm_args)
    audio_lm_args.grad_cp = 0
    audio_lm_args.dropout = 0
    print(audio_lm_args)
    audio_lm_model = RWKV7ModelForLatentInputsCuda(audio_lm_args)
    print(audio_lm_model)
    audio_lm_model = audio_lm_model.to(device)
    print(audio_lm_model)
    asr_model = RWKV7ASRModelCuda(whisper_encoder, audio_lm_model, llm, whisper_feature_extractor)
    
    ckpt_file = args.ckpt_file
    info = asr_model.load_state_dict(torch.load(ckpt_file, map_location='cpu'))
    print(info)
    
    print(asr_model)
    dtype = torch.bfloat16
    asr_model = asr_model.to(dtype).to(device)
    tokenizer_path = os.path.join(model_dir, "rwkv_vocab_v20230424.txt")
    tokenizer = TRIE_TOKENIZER(tokenizer_path)
    print(tokenizer)
    instruction_input_ids = tokenizer.encode(instruction)
    hints_input_ids = tokenizer.encode(hints)
    instruction_input_ids = torch.tensor(instruction_input_ids, dtype=torch.long,device=device).unsqueeze(0)
    hints_input_ids = torch.tensor(hints_input_ids, dtype=torch.long,device=device).unsqueeze(0)
    audio_path = audio_path
    audio_data,sample_rate = sf.read(audio_path)
    if sample_rate != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
    
    print(f'type of audio_data: {type(audio_data)}')
    print(f'shape of audio_data: {audio_data.shape}')
    print(f'sample_rate: {sample_rate}')

    print(f'instruction_input_ids: {instruction_input_ids}')
    print(f'hints_input_ids: {hints_input_ids}')
    print(f'audio_data: {audio_data}')
    instruction_attention_mask = torch.ones(instruction_input_ids.shape[0], instruction_input_ids.shape[1],dtype=torch.long).to(instruction_input_ids.device)
    print(f'instruction_attention_mask: {instruction_attention_mask}')
    result_tokens = asr_model.forward_inference([audio_data], instruction_input_ids, instruction_attention_mask, hints_input_ids)
    print(f'result_tokens: {result_tokens}')
