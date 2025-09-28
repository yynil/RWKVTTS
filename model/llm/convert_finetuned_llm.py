from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import argparse
import torch
import shutil
import os
import json
import glob

args = argparse.ArgumentParser()
args.add_argument('--model_path', type=str, required=True)
args.add_argument('--checkpoint_path', type=str, required=True)
args.add_argument('--output_path', type=str, required=True)
args = args.parse_args()

print(f'Loading model from {args.model_path}')
model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
print(f'Loading checkpoint from {args.checkpoint_path}')

# Check if checkpoint_path is a directory or file
if os.path.isdir(args.checkpoint_path):
    # Check for pytorch_model.bin.index.json file
    index_file = os.path.join(args.checkpoint_path, 'pytorch_model.bin.index.json')
    if os.path.exists(index_file):
        print(f'Found index file: {index_file}')
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        # Load state dict from multiple files
        state_dict = {}
        weight_map = index_data.get('weight_map', {})
        
        # Get unique model files from weight_map
        model_files = set(weight_map.values())
        print(f'Loading from {len(model_files)} model files: {model_files}')
        
        for model_file in model_files:
            file_path = os.path.join(args.checkpoint_path, model_file)
            print(f'Loading weights from: {file_path}')
            checkpoint = torch.load(file_path, map_location='cpu')
            
            # Filter weights that belong to this file
            for key, value in checkpoint.items():
                if weight_map.get(key) == model_file:
                    state_dict[key] = value
        
        print(f'Loaded {len(state_dict)} parameter groups')
        model.load_state_dict(state_dict)
    else:
        # Try to find pytorch_model.bin files directly
        model_files = glob.glob(os.path.join(args.checkpoint_path, 'pytorch_model-*.bin'))
        if model_files:
            print(f'Found model files: {model_files}')
            state_dict = {}
            for model_file in sorted(model_files):
                print(f'Loading weights from: {model_file}')
                checkpoint = torch.load(model_file, map_location='cpu')
                state_dict.update(checkpoint)
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f'No pytorch_model.bin.index.json or pytorch_model-*.bin files found in {args.checkpoint_path}')
else:
    # Single file checkpoint
    model.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'))

print(f'Saving model to {args.output_path}')
model.save_pretrained(args.output_path)

print(f'Loading tokenizer from {args.model_path}')
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
print(f'Saving tokenizer to {args.output_path}')
tokenizer.save_pretrained(args.output_path)

print(f'Copying *.py files, *.txt files to {args.output_path}')
# Copy *.py files, *.txt files to output_path
for pattern in ['*.py', '*.txt']:
    source_files = glob.glob(os.path.join(args.model_path, pattern))
    for source_file in source_files:
        filename = os.path.basename(source_file)
        dest_file = os.path.join(args.output_path, filename)
        shutil.copy2(source_file, dest_file)
        print(f'Copied {filename} to {args.output_path}')