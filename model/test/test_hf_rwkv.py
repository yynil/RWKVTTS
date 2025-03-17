model_path = "/external_data/models/rwkv7-0.4B-world/"
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
device = 'cuda:0'
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device=device, dtype=torch.float16)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
prompt = "User : 请评价一下现阶段中美关系，包括合作和竞争。\nAssistant :"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
print(inputs)
from transformers import GenerationConfig
config = GenerationConfig(max_new_tokens=256)

ids = tokenizer.encode(prompt,add_special_tokens=False)
print(ids)

outputs = model.generate(**inputs, 
                         generation_config=config,)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# from fla.models.rwkv7 import RWKV7ForCausalLM, RWKV7Model, RWKV7Config