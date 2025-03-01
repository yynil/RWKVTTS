model_path = "/home/yueyulin/models/rwkv7-2.9B-world"
from tracemalloc import stop
from regex import F
from transformers import AutoModelForCausalLM, AutoTokenizer
device = 'cuda:2'
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half()
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
prompt = "User : 请评价一下现阶段中美关系，包括合作和竞争。\nAssistant :"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
print(inputs)
from transformers import GenerationConfig
config = GenerationConfig(max_new_tokens=256)

ids = tokenizer.encode(prompt,add_special_tokens=False)
print(ids)
# outputs = model.generate(**inputs, 
#                          generation_config=config,)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# from fla.models.rwkv7 import RWKV7ForCausalLM, RWKV7Model, RWKV7Config