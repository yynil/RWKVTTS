
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