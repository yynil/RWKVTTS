import argparse
import os
import json
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Model,RWKV7Config
from transformers import AutoModelForCausalLM
from model.llm.rwkv_asr import RWKV7ASRModel
import torch
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--llm_model_path", type=str, required=True)
    args = parser.parse_args()
    config_file = os.path.join(args.model_path, "config.json")
    with open(config_file, "r") as f:
        config = json.load(f)
    print(config)
    rwkv_config = RWKV7Config.from_dict(config)
    print(rwkv_config)
    audio_lm_model = RWKV7Model(rwkv_config)
    print(audio_lm_model)
    llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model_path, trust_remote_code=True)
    print(llm_model)
    asr_audio_lm_model = RWKV7ASRModel(audio_lm_model, llm_model)   
    print(asr_audio_lm_model)

    info = asr_audio_lm_model.load_state_dict(torch.load(args.ckpt_path),strict=False)
    #check if missing_keys are started with "llm." and unexpected_keys is empty
    assert all(key.startswith("llm.") for key in info.missing_keys), "missing_keys are not started with 'llm.'"
    assert len(info.unexpected_keys) == 0, "unexpected_keys is not empty"
    print("✅check passed")

    #mkdir output_path/llm_model and copy llm_model_path/* to output_path/llm_model
    print(f"🔄copying {args.llm_model_path} to {os.path.join(args.output_path, 'llm_model')}")
    os.makedirs(os.path.join(args.output_path, "llm_model"), exist_ok=True)
    for file in os.listdir(args.llm_model_path):
        src_path = os.path.join(args.llm_model_path, file)
        dst_path = os.path.join(args.output_path, "llm_model", file)
        if os.path.isfile(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"跳过目录: {file}")
    print(f"✅Finished copying {args.llm_model_path} to {os.path.join(args.output_path, 'llm_model')}")

    #mkdir output_path/audio_lm_model and copy model_path/*config to output_path/audio_lm_model
    print(f"🔄copying {args.model_path} to {os.path.join(args.output_path, 'audio_lm_model')}")
    os.makedirs(os.path.join(args.output_path, "audio_lm_model"), exist_ok=True)
    shutil.copy(os.path.join(args.model_path, "config.json"), os.path.join(args.output_path, "audio_lm_model", "config.json"))
    print(f"✅Finished copying {args.model_path} to {os.path.join(args.output_path, 'audio_lm_model')}")

    #save ckpt to output_path
    print(f"🔄saving {args.ckpt_path} to {args.output_path}")
    shutil.copy(args.ckpt_path, os.path.join(args.output_path, "asr_audio_lm_model.pt"))
    print(f"✅Finished saving {args.ckpt_path} to {args.output_path}")
