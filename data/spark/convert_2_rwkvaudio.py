from model.llm.spark_llm import RWKV7SpeechConfig,RWKV7ForSpeech
from transformers import AutoConfig
from safetensors.torch import load_file
import glob
import os
import shutil
def load_multiple_safetensors(model_path):
    """加载多个 safetensors 文件并合并它们
    
    Args:
        model_path: 包含 safetensors 文件的目录路径
        
    Returns:
        合并后的 state dict
    """
    # 获取所有 safetensors 文件
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {model_path}")
    
    # 创建合并的 state dict
    merged_state_dict = {}
    
    # 加载并合并每个文件
    for file_path in safetensors_files:
        print(f"Loading {file_path}")
        state_dict = load_file(file_path)
        merged_state_dict.update(state_dict)
    
    return merged_state_dict
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/home/yueyulin/models/rwkv7-0.1B-g1")
    parser.add_argument("--output_dir", type=str, default="/home/yueyulin/models/rwkv7-0.1B-g1-respark-speech")
    parser.add_argument("--audio_vocab_size", type=int, default=8193)
    parser.add_argument("--text_vocab_size", type=int, default=65536)
    parser.add_argument("--audio_global_vocab_size", type=int, default=4096)
    args = parser.parse_args()

    state_dict = load_multiple_safetensors(args.model_dir)
    original_config = AutoConfig.from_pretrained(args.model_dir)
    print(original_config)
    args.text_vocab_size = original_config.vocab_size
    kwargs = original_config.to_dict()
    kwargs.pop('vocab_size')
    new_config = RWKV7SpeechConfig(
        vocab_size=args.audio_vocab_size,
        text_vocab_size=args.text_vocab_size,
        audio_global_vocab_size=args.audio_global_vocab_size,
        **kwargs
    )
    new_config.architectures = ['RWKV7ForSpeech']
    new_config.auto_map = {
        "AutoConfig": "modeling_rwkvspeech.RWKV7SpeechConfig",
        "AutoModel": "modeling_rwkvspeech.RWKV7Model",
        "AutoModelForCausalLM": "modeling_rwkvspeech.RWKV7ForSpeech"
    }
    print(new_config)

    rwkv7_speech_model = RWKV7ForSpeech._from_config(new_config)
    rwkv7_speech_model.copy_state_dict(state_dict)
    print(rwkv7_speech_model)

    os.makedirs(args.output_dir,exist_ok=True)
    rwkv7_speech_model.save_pretrained(args.output_dir)

    #copy modeling_rwkvspeech.py to output_dir
    shutil.copy(os.path.join(os.path.dirname(__file__), "modeling_rwkvspeech.py"), args.output_dir)
    print(f"copy modeling_rwkvspeech.py to {args.output_dir}")

    # copy hf_rwkv_tokenizer.py from model_dir to output_dir
    shutil.copy(os.path.join(args.model_dir, "hf_rwkv_tokenizer.py"), args.output_dir)
    print(f"copy hf_rwkv_tokenizer.py to {args.output_dir}")

    # copy *txt from model_dir to output_dir
    for file in os.listdir(args.model_dir):
        if file.endswith('.txt'):
            shutil.copy(os.path.join(args.model_dir, file), os.path.join(args.output_dir, file))
            print(f"Copied {file} to {args.output_dir}")

    #copy tokenizer_config.json special_tokens_map.json added_tokens.json to output_dir
    shutil.copy(os.path.join(args.model_dir, "tokenizer_config.json"), args.output_dir)
    print(f"copy tokenizer_config.json to {args.output_dir}")
    shutil.copy(os.path.join(args.model_dir, "special_tokens_map.json"), args.output_dir)
    print(f"copy special_tokens_map.json to {args.output_dir}")
    shutil.copy(os.path.join(args.model_dir, "added_tokens.json"), args.output_dir)
    print(f"copy added_tokens.json to {args.output_dir}")