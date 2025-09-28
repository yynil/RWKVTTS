#!/usr/bin/env python3
"""
Test script to verify RWKV7XYLM model can handle the data format correctly
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.llm.xy_llm import RWKV7XYLM, RWKV7XYConfig
from data.utils.collator import xy_data_collator
from data.spark.multiple_webdataset import MultipleWebDataset
from transformers import AutoTokenizer
from XY_Tokenizer.xy_tokenizer.model import XY_Tokenizer

def test_xy_model():
    """Test the RWKV7XYLM model with our data format"""
    
    # Configuration
    model_path = "/home/yueyulin/models/rwkv7-xy-0.4B-g1/"
    xy_config_path = "third_party/XY_Tokenizer/config/xy_tokenizer_config.yaml"
    xy_ckpt_path = "/home/yueyulin/models/XY_Tokenizer_TTSD_V0/xy_tokenizer.ckpt"
    data_dir = "/external_data/voxbox_wids/aishell-3/aishell-3_0000.tar"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("Loading text tokenizer...")
    text_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print("Loading XY_Tokenizer...")
    xy_tokenizer = XY_Tokenizer.load_from_checkpoint(xy_config_path, xy_ckpt_path)
    xy_tokenizer.eval().to(device)
    
    print("Loading model...")
    model = RWKV7XYLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    )
    model.zero_embs()
    model.to(device)
    num_channels = model.config.num_channels
    print(f"Model configured with {num_channels} channels")
    print(f"Text vocab size: {model.config.vocab_size}")
    print(f"Speech vocab size: {model.config.speech_vocab_size}")
    
    print("Loading dataset...")
    dataset = MultipleWebDataset(
        data_dir=data_dir,
        target_sr=16000,
        target_channels=1,
        shuffle=False,
        verify_tar=False
    )
    
    print("Creating data collator...")
    text_shift_size = 65536
    speech_vocab_size = model.config.speech_vocab_size
    data_collator = lambda features: xy_data_collator(
        features, text_tokenizer, xy_tokenizer, num_channels, text_shift_size, speech_vocab_size, device
    )
    
    print("Testing with a single batch...")
    batch = data_collator([dataset[0], dataset[1]])
    
    if not batch:
        print("Error: Empty batch!")
        return
    
    print(f"Batch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    
    print(f"Input IDs value ranges:")
    for i in range(num_channels):
        channel_data = batch['input_ids'][:, :, i]
        min_val = channel_data.min().item()
        max_val = channel_data.max().item()
        print(f"  Channel {i}: {min_val} to {max_val}")
    
    print(f"Labels value ranges:")
    for i in range(num_channels):
        channel_data = batch['labels'][:, :, i]
        min_val = channel_data.min().item()
        max_val = channel_data.max().item()
        print(f"  Channel {i}: {min_val} to {max_val}")
    
    print("Running model forward pass...")
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                labels=batch['labels'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                return_dict=True,
                use_cache=False
            )
            
            print("✓ Model forward pass successful!")
            print(f"Loss: {outputs.loss.item()}")
            print(f"Logits shapes:")
            for i, logits in enumerate(outputs.logits):
                print(f"  Channel {i}: {logits.shape}")
                
        except Exception as e:
            print(f"✗ Model forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    print("✓ All tests passed!")

if __name__ == "__main__":
    test_xy_model() 