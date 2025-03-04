
from transformers import AutoTokenizer
def get_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return tokenizer