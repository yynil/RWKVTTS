#!/usr/bin/env python3
"""
Test script to verify the training setup for RWKV7XYLM
"""

import os
import sys
import torch
from transformers import AutoTokenizer
from functools import partial
import glob

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.llm.xy_llm import RWKV7XYLM
from data.spark.multiple_webdataset import MultipleWebDataset
from data.utils.collator import xy_data_collator
from XY_Tokenizer.xy_tokenizer.model import XY_Tokenizer

def test_training_setup():
    """Test the training setup components"""
    
    # Configuration
    model_path = "/home/yueyulin/models/rwkv7-xy-0.4B-g1/"
    xy_config_path = "third_party/XY_Tokenizer/config/xy_tokenizer_config.yaml"
    xy_ckpt_path = "/home/yueyulin/models/XY_Tokenizer_TTSD_V0/xy_tokenizer.ckpt"
    data_dir = "/external_data/voxbox_wids/aishell-3/"
    
    print("Testing training setup...")
    
    # 1. Test model loading
    print("1. Loading model...")
    try:
        model = RWKV7XYLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        num_channels = model.config.num_channels
        print(f"   ✓ Model loaded successfully with {num_channels} channels")
    except Exception as e:
        print(f"   ✗ Model loading failed: {e}")
        return False
    
    # 2. Test tokenizer loading
    print("2. Loading tokenizer...")
    try:
        text_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("   ✓ Text tokenizer loaded successfully")
    except Exception as e:
        print(f"   ✗ Text tokenizer loading failed: {e}")
        return False
    
    # 3. Test XY_Tokenizer loading
    print("3. Loading XY_Tokenizer...")
    try:
        xy_tokenizer = XY_Tokenizer.load_from_checkpoint(xy_config_path, xy_ckpt_path)
        xy_tokenizer.eval().to('cuda')
        print("   ✓ XY_Tokenizer loaded successfully")
    except Exception as e:
        print(f"   ✗ XY_Tokenizer loading failed: {e}")
        return False
    
    # 4. Test dataset loading
    print("4. Loading dataset...")
    try:
        data_files = glob.glob(os.path.join(data_dir, "*.tar"))
        if not data_files:
            print(f"   ✗ No .tar files found in {data_dir}")
            return False
        
        dataset = MultipleWebDataset(
            data_files=data_files,
            target_sr=16000,
            target_channels=1,
            shuffle=True,
            verify_tar=False
        )
        print(f"   ✓ Dataset loaded successfully with {len(data_files)} files")
    except Exception as e:
        print(f"   ✗ Dataset loading failed: {e}")
        return False
    
    # 5. Test data collator
    print("5. Testing data collator...")
    try:
        data_collator = partial(
            xy_data_collator,
            text_tokenizer=text_tokenizer,
            xy_tokenizer=xy_tokenizer,
            num_channels=num_channels,
            device=torch.device('cuda')
        )
        
        # Test with a small batch
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=data_collator)
        test_batch = next(iter(dataloader))
        
        if test_batch:
            print(f"   ✓ Data collator works - batch shapes:")
            print(f"     input_ids: {test_batch['input_ids'].shape}")
            print(f"     labels: {test_batch['labels'].shape}")
            print(f"     attention_mask: {test_batch['attention_mask'].shape}")
        else:
            print("   ✗ Data collator returned empty batch")
            return False
    except Exception as e:
        print(f"   ✗ Data collator test failed: {e}")
        return False
    
    # 6. Test model forward pass
    print("6. Testing model forward pass...")
    try:
        model.to('cuda')
        model.eval()
        
        with torch.no_grad():
            outputs = model(
                input_ids=test_batch['input_ids'].to('cuda'),
                labels=test_batch['labels'].to('cuda'),
                attention_mask=test_batch['attention_mask'].to('cuda')
            )
        
        print(f"   ✓ Model forward pass successful - loss: {outputs.loss.item():.4f}")
    except Exception as e:
        print(f"   ✗ Model forward pass failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Training setup is ready.")
    return True

if __name__ == "__main__":
    success = test_training_setup()
    sys.exit(0 if success else 1) 