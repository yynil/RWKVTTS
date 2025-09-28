from model.llm.rwkv_asr_cuda_whisper import load_whisper_feature_extractor_and_encoder
import os
import shutil
import torch
import argparse
from utils.rwkv_utilities import parser_config_from_checkpoint
from model.llm.rwkv_asr_cuda_whisper import RWKV7ModelForCausalLMCuda,RWKV7ModelForLatentInputsCuda
from argparse import Namespace
import questionary
import json
def convert_args_to_json(args):
    return json.dumps(args.__dict__)
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--llm_ckpt_file", type=str, required=True)
    parser.add_argument("--whisper_path", type=str, required=True)
    input_args = parser.parse_args()

    os.makedirs(input_args.output_path, exist_ok=True)

    args = parser_config_from_checkpoint(input_args.llm_ckpt_file)
    print(args)
    llm = RWKV7ModelForCausalLMCuda(args)
    print(llm)  
    llm_args = convert_args_to_json(args)
    print(llm_args)
    llm_args_path = os.path.join(input_args.output_path, 'llm_args.json')
    print(f'save llm_args to {llm_args_path}')
    with open(llm_args_path, 'w') as f:
        f.write(llm_args)
    llm_ckpt_path = os.path.join(input_args.output_path, 'llm_state_dict.pt')
    print(f'save llm state_dict to {llm_ckpt_path}')
    torch.save(llm.state_dict(), llm_ckpt_path)
    whisper_feature_extractor, whisper_encoder = load_whisper_feature_extractor_and_encoder(input_args.whisper_path)
    print(whisper_feature_extractor)
    print(whisper_encoder)
    print('copy whisper encoder and feature extractor to output_path')
    os.makedirs(os.path.join(input_args.output_path, 'whisper_encoder'), exist_ok=True)
    shutil.copy(os.path.join(input_args.whisper_path, 'config.json'), os.path.join(input_args.output_path, 'whisper_encoder', 'config.json'))
    shutil.copy(os.path.join(input_args.whisper_path, 'preprocessor_config.json'), os.path.join(input_args.output_path, 'whisper_encoder', 'preprocessor_config.json'))
    torch.save(whisper_encoder.state_dict(), os.path.join(input_args.output_path, 'whisper_encoder', 'pytorch_model.bin'))
    #Namespace(n_head=12, head_size=64, head_size_a=64, head_size_divisor=1, vocab_size=65536, n_embd=768, n_layer=12, need_init_tmix=False, need_init_cmix=False, dropout=0, grad_cp=0)
    audio_lm_args = Namespace()
    n_head = questionary.text('n_head', default='12',validate=lambda x: x.isdigit() and int(x) > 0).ask()
    audio_lm_args.n_head = int(n_head)
    head_size = questionary.text('head_size', default='64',validate=lambda x: x.isdigit() and int(x) > 0).ask()
    audio_lm_args.head_size = int(head_size)
    audio_lm_args.head_size_a = int(head_size)
    n_embd = questionary.text('n_embd', default='768',validate=lambda x: x.isdigit() and int(x) > 0).ask()
    audio_lm_args.n_embd = int(n_embd)
    n_layer = questionary.text('n_layer', default='12',validate=lambda x: x.isdigit() and int(x) > 0).ask()
    audio_lm_args.n_layer = int(n_layer)

    
    audio_lm_args.head_size_divisor = 1
    audio_lm_args.need_init_tmix = True
    audio_lm_args.need_init_cmix = True
    audio_lm_args.dropout = 0
    audio_lm_args.grad_cp = 0
    print(audio_lm_args)
    audio_lm_args_path = os.path.join(input_args.output_path, 'audio_lm_args.json')
    print(f'save audio_lm_args to {audio_lm_args_path}')
    with open(audio_lm_args_path, 'w') as f:
        f.write(convert_args_to_json(audio_lm_args))
    audio_lm_ckpt_path = os.path.join(input_args.output_path, 'audio_lm_state_dict.pt')
    print(f'save audio_lm state_dict to {audio_lm_ckpt_path}')
    audio_lm = RWKV7ModelForLatentInputsCuda(audio_lm_args)
    torch.save(audio_lm.state_dict(), audio_lm_ckpt_path)
    print(f'done')