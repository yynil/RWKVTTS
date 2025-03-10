import torch
import sys
original_pt_file = sys.argv[1]
output_pt_file = sys.argv[2]
print(f'Converting {original_pt_file} to {output_pt_file}')
#load the orginal model and convert all fp32 parameters to bf16
print(f'Loading {original_pt_file}')
model = torch.load(original_pt_file)
new_states = {}
for k,v in model.items():
    if v.dtype == torch.float32:
        v = v.bfloat16()
    new_states[k] = v
print(f'Saving {output_pt_file}')
torch.save(new_states,output_pt_file)
print(f'Finished converting {original_pt_file} to {output_pt_file}')