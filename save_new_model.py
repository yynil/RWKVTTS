import argparse
import torch
import os
import shutil
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--ckpt_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)

args = parser.parse_args()

print(args)
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(args.model_path,trust_remote_code=True)
print(f"model loaded from {args.model_path}")
model.load_state_dict(torch.load(args.ckpt_path))
print(f"ckpt loaded from {args.ckpt_path}")

model.save_pretrained(args.output_path)
print(f"model saved to {args.output_path}")


#copy model_path/*txt to output_path
print(f"copying model_path/*txt to output_path")
for file in os.listdir(args.model_path):
    if file.endswith(".txt"):
        shutil.copy(os.path.join(args.model_path, file), os.path.join(args.output_path, file))

#copy model_path/*json to output_path
print(f"copying model_path/*json to output_path")
for file in os.listdir(args.model_path):
    if file.endswith(".json"):
        shutil.copy(os.path.join(args.model_path, file), os.path.join(args.output_path, file))

#copy model_path/*py to output_path
print(f"copying model_path/*py to output_path")
for file in os.listdir(args.model_path):
    if file.endswith(".py"):
        shutil.copy(os.path.join(args.model_path, file), os.path.join(args.output_path, file))