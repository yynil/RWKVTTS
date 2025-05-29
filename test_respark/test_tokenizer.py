rwkv_model_dir = '/home/yueyulin/models/rwkv7-0.1B-g1'
from transformers import AutoTokenizer
from utils.utilities import get_respark_tts_tokenizer
tokenizer = get_respark_tts_tokenizer(rwkv_model_dir)

original_tokenizer = AutoTokenizer.from_pretrained(rwkv_model_dir, trust_remote_code=True)

print(tokenizer.vocab_size)
print(original_tokenizer.vocab_size)

print(tokenizer.encode("<tts>"))