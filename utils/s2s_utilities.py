import os
import re
from langdetect import detect, LangDetectException
import torch
from torch.nn.utils.rnn import pad_sequence

CHINESE_ASR_TEMPLATE = "User:请把以下语音转化成中文文本。\n{SEMANTICS}\nAssistant:"
ENGLISH_ASR_TEMPLATE = "User:Please convert the following audio to English text.\n{SEMANTICS}\nAssistant:"
UNIVERSAL_ASR_TEMPLATE = "User:Please convert the following audio to text.\n{SEMANTICS}\nAssistant:"

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return 'en'

def create_asr_inputs_and_labels(batch, tokenizer, eos_token_id, pad_token_id):
    """简化的ASR输入创建函数，只返回3个值，去掉attention_mask_to_mask_audio_tokens"""
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    
    for item in batch:
        semantic_tokens = item['semantic_tokens']
        text = item['text']
        
        language = detect_language(text)
        if language.startswith('zh'):
            template = CHINESE_ASR_TEMPLATE
        else:
            template = ENGLISH_ASR_TEMPLATE
        
        # 只使用semantic tokens，忽略global tokens
        semantic_str = ''.join([f'SEMANTIC_TOKEN_ID_{token_id}' for token_id in semantic_tokens])
        prompt = template.format(SEMANTICS=semantic_str)
        
        prompt_ids = tokenizer.encode(prompt)
        output_ids = tokenizer.encode(text)
        input_ids = prompt_ids + output_ids + [eos_token_id]
        labels = [-100] * len(prompt_ids) + output_ids + [eos_token_id]
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        
        # 创建attention mask
        full_attention_mask = torch.ones(len(input_ids), dtype=torch.long)
        attention_mask_list.append(full_attention_mask)
    
    return pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id, padding_side='left'), \
           pad_sequence(labels_list, batch_first=True, padding_value=pad_token_id, padding_side='left'), \
           pad_sequence(attention_mask_list, batch_first=True, padding_value=0, padding_side='left')

if __name__ == "__main__":
    data_dir = '/home/yueyulin/data/voxbox_wids_tokens_filtered/aishell-3'
    from datasets import load_dataset
    import glob
    jsonl_files = glob.glob(os.path.join(data_dir, '*.jsonl'))
    dataset = load_dataset("json", data_files=jsonl_files, split="train")

    def simple_collate_fn(batch):
        return batch
    
    tokenizer_file = '/home/yueyulin/models/rwkvs2s/rwkv_vocab_enlarged.txt'
    from tokenizer.rwkv_tokenizer import RWKV_TOKENIZER

    tokenizer = RWKV_TOKENIZER(tokenizer_file)
    print(tokenizer.encode('Assistant:'))
    print(tokenizer.encode('User:'))
    eos_token_id = 0
    pad_token_id = 0
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=simple_collate_fn)
    for batch in dataloader:
        print(batch)
        input_ids, labels, attention_mask = create_asr_inputs_and_labels(batch, tokenizer, eos_token_id, pad_token_id)
        print(input_ids.tolist())
        print(labels.tolist())
        print(attention_mask.tolist())
        break

