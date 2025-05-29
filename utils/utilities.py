import os
import json
from transformers import AutoTokenizer
def get_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    special_tokens = {
            'pad_token': '<|rwkv_tokenizer_end_of_text|>',
            'additional_special_tokens': [
                '<|endofprompt|>',
                '[breath]', '<strong>', '</strong>', '[noise]',
                '[laughter]', '[cough]', '[clucking]', '[accent]',
                '[quick_breath]',
                "<laughter>", "</laughter>",
                "[hissing]", "[sigh]", "[vocalized-noise]",
                "[lipsmack]", "[mn]"
            ]
        }
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer

def get_respark_tts_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    original_vocab_size = tokenizer.vocab_size
    added_tokens_file = os.path.join(os.path.dirname(__file__),'spark_tts_added_tokens.json')
    with open(added_tokens_file, 'r') as f:
        added_tokens = json.load(f)
    tokenizer.add_special_tokens(added_tokens)
    return tokenizer,original_vocab_size