import os
import torch
import torch.nn as nn
import shutil
import argparse
from transformers import AutoTokenizer
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM
from xy_llm import RWKV7XYLM, RWKV7XYConfig

def convert_model(source_path, dest_path, num_channels, speech_vocab_size):
    """
    Converts a standard RWKV7 model to a multi-channel RWKV7XYLM model.
    """
    print(f"Loading original model from {source_path}")
    # Add trust_remote_code=True for custom models
    original_model = RWKV7ForCausalLM.from_pretrained(source_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(source_path, trust_remote_code=True)

    original_vocab_size = original_model.config.vocab_size
    print(f"Original vocab size: {original_vocab_size}")

    # 1. Extend tokenizer with special and speech tokens
    special_tokens = [f"[S{i}]" for i in range(10)] + [f"[CTL{i}]" for i in range(90)]
    # The speech tokens are added to the main vocabulary for the text channel (channel 0)
    # but their size is determined by speech_vocab_size for the speech channels.
    speech_tokens_for_tokenizer = [f"[SP{i}]" for i in range(speech_vocab_size)]
    
    num_added_toks = tokenizer.add_tokens(speech_tokens_for_tokenizer+special_tokens)
    print(f"Added {num_added_toks} new tokens to the tokenizer.")
    
    new_vocab_size = len(tokenizer)
    print(f"New vocab size for channel 0: {new_vocab_size}")

    # 2. Create new model configuration
    config_dict = original_model.config.to_dict()
    config_dict['num_channels'] = num_channels
    config_dict['vocab_size'] = new_vocab_size
    config_dict['speech_vocab_size'] = speech_vocab_size
    new_config = RWKV7XYConfig(**config_dict)
    new_config.architectures = ['RWKV7ForMultiCodebooksSpeech']
    new_config.auto_map = {
        "AutoConfig": "xy_llm.RWKV7XYConfig",
        "AutoModel": "xy_llm.RWKV7XYLM",
        "AutoModelForCausalLM": "xy_llm.RWKV7XYLM"
    }
    # 3. Instantiate the new RWKV7XYLM model
    print("Instantiating new RWKV7XYLM model")
    new_model = RWKV7XYLM(new_config)
    
    # 4. Copy weights
    print("Copying model weights...")
    # Copy backbone weights, excluding the embedding layer which is handled separately
    original_model_state_dict = original_model.model.state_dict()
    original_model_state_dict.pop('embeddings.weight', None) # Avoid size mismatch
    new_model.model.load_state_dict(original_model_state_dict, strict=False)

    with torch.no_grad():
        # Handle Channel 0 (text) emb and head
        print("Processing weights for Channel 0 (text)...")
        # Copy overlapping embedding weights
        new_model.embs[0].weight[:original_vocab_size, :] = original_model.model.embeddings.weight[:original_vocab_size, :]
        # Initialize new embedding weights
        nn.init.normal_(new_model.embs[0].weight[original_vocab_size:, :-1], mean=0.0, std=new_model.config.initializer_range)
        
        # Copy overlapping head weights
        new_model.heads[0].weight[:original_vocab_size, :] = original_model.lm_head.weight[:original_vocab_size, :]
        # Initialize new head weights
        nn.init.normal_(new_model.heads[0].weight[original_vocab_size:, :], mean=0.0, std=new_model.config.initializer_range)
        if hasattr(original_model.lm_head, 'bias') and original_model.lm_head.bias is not None:
            new_model.heads[0].bias.data[:original_vocab_size] = original_model.lm_head.bias.data[:original_vocab_size]
            nn.init.zeros_(new_model.heads[0].bias.data[original_vocab_size:])

        # Handle other channels (speech)
        for i in range(1, new_config.num_channels):
            print(f"Initializing weights for Channel {i} (speech)...")
            nn.init.normal_(new_model.embs[i].weight, mean=0.0, std=new_model.config.initializer_range)
            nn.init.normal_(new_model.heads[i].weight, mean=0.0, std=new_model.config.initializer_range)
            if new_model.heads[i].bias is not None:
                nn.init.zeros_(new_model.heads[i].bias)

    print("Weight conversion complete.")

    # 5. Save the new model, tokenizer, and necessary files
    print(f"Saving new model and tokenizer to {dest_path}")
    os.makedirs(dest_path, exist_ok=True)
    
    new_model.save_pretrained(dest_path)
    tokenizer.save_pretrained(dest_path)
    new_model.zero_embs()
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "xy_llm.py"),
        os.path.join(dest_path, "xy_llm.py")
    )
    
    for filename in os.listdir(source_path):
        if filename.endswith(".txt"):
            shutil.copyfile(
                os.path.join(source_path, filename),
                os.path.join(dest_path, filename)
            )
            print(f"Copied {filename}")

    print("Conversion process finished successfully.")

def main():
    parser = argparse.ArgumentParser(description="Convert a standard RWKV7 model to a multi-channel RWKV7XYLM model.")
    parser.add_argument("--source_path", type=str, default="/home/yueyulin/models/rwkv7-0.4B-g1", help="Path to the source RWKV7 model directory.")
    parser.add_argument("--dest_path", type=str, default="/home/yueyulin/models/rwkv7-xy-0.4B-g1", help="Path to save the new RWKV7XYLM model.")
    parser.add_argument("--num_channels", type=int, default=8, help="Number of channels for the new model (1 text + N speech). Default 9 for 8 speech channels.")
    parser.add_argument("--speech_vocab_size", type=int, default=1025, help="Vocabulary size for the speech channels.")
    
    args = parser.parse_args()

    if not os.path.exists(args.source_path):
        print(f"Error: Source model path does not exist: {args.source_path}")
        print("Please ensure the original model is available at the specified path.")
        return

    convert_model(args.source_path, args.dest_path, args.num_channels, args.speech_vocab_size)

if __name__ == '__main__':
    main()