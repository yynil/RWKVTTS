import datasets
import os
import json
import torch
import random
import time
random.seed(time.time())
def load_jsonl_dataset(directory,tokenizer):
    '''
    load jsonl files in a directory recursively
    '''
    data_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jsonl'):
                data_files.append(os.path.join(root, file))
    dataset = datasets.load_dataset('json', data_files=data_files)['train']
    #1. concatenate llm_prompt_speech_token and tts_speech_tokens (list of int)
    #delay the concatenation to collate_fn since sometimes we want to drop the prompt
    # dataset = dataset.map(lambda x: {'speech_token': x['llm_prompt_speech_token'] + x['tts_speech_tokens']},remove_columns=['tts_speech_tokens','llm_prompt_speech_token'])
    #2. Filter the data either :
    #   1. the length of the speech_token is less than 1
    #   2. the length of the speech_token is greater than 1000
    #   3. the length of the text is greater than 500
    #   4. the length of the prompt_text is greater than 500
    #   5. the length of the text_token is less than 1
    #   6. the length of the prompt_text_token is less than 1
    dataset = dataset.filter(lambda x: len(x['llm_prompt_speech_token']) > 1 and len(x['llm_prompt_speech_token']) < 1000 and len(x['tts_speech_tokens']) > 1 and len(x['tts_speech_tokens']) < 1000
                             and len(tokenizer.encode(x['text'])) < 500 and len(tokenizer.encode(x['prompt_text'])) < 500 and len(tokenizer.encode(x['text'])) > 1 and len(tokenizer.encode(x['prompt_text'])) > 1)
    #2. tokenize the text to text_tokens and prompt_text to prompt_text_tokens
    # dataset = dataset.map(lambda x: {'text_tokens': tokenizer.encode(x['text']), 'prompt_text_tokens': tokenizer.encode(x['prompt_text'])},remove_columns=['text','prompt_text'])
    return dataset

def collate_fn(batch, tokenizer, pad_to_max_length=True, max_length=2048, drop_prompt_audio_rate=-0.1):
    '''
    convert the data to torch tensors
    1. call tokenizer.encode('text') and tokenizer.encode('prompt_text'), concatenate them to get the text_token, record each sample's length to text_token_len
    2. convert the text_tokens and text_token_len to torch tensor
    3. record each sample's speech_token length to speech_token_len
    4. convert the speech_token and speech_token_len to torch tensor
    5. We will drop prompt with drop_prompt_audio_rate to ask model to learn generate audio without guaidance
    By default we won't drop anything
    '''
    all_text_tokens = []
    all_speech_tokens = []
    speech_token_len = []
    text_token_len = []
    my_max_length = 0
    is_drop_prompt = random.random() < drop_prompt_audio_rate
    
    for sample in batch:
        tts_speech_tokens = sample['tts_speech_tokens']
        llm_prompt_speech_token = sample['llm_prompt_speech_token']
        
        if is_drop_prompt:
            # 只使用文本部分，不使用提示
            text_tokens = tokenizer.encode(sample['text'])
            all_text_tokens.append(torch.tensor(text_tokens, dtype=torch.int32))
            text_token_len.append(len(text_tokens))
            
            # 只使用语音部分，不使用提示语音
            current_speech_tokens = tts_speech_tokens
            all_speech_tokens.append(torch.tensor(current_speech_tokens, dtype=torch.int32))
            speech_token_len.append(len(current_speech_tokens))
            
            total_length = len(text_tokens) + len(current_speech_tokens)
        else:
            # 使用提示+文本
            text_tokens = tokenizer.encode(sample['text'])
            prompt_tokens = tokenizer.encode(sample['prompt_text'])
            combined_text_tokens = prompt_tokens + text_tokens
            all_text_tokens.append(torch.tensor(combined_text_tokens, dtype=torch.int32))
            text_token_len.append(len(combined_text_tokens))
            
            # 使用提示语音+语音
            current_speech_tokens = llm_prompt_speech_token + tts_speech_tokens
            all_speech_tokens.append(torch.tensor(current_speech_tokens, dtype=torch.int32))
            speech_token_len.append(len(current_speech_tokens))
            
            total_length = len(combined_text_tokens) + len(current_speech_tokens)
            
        if total_length > my_max_length:
            my_max_length = total_length
    
    # 检查长度是否超出最大长度
    skip = my_max_length > max_length
    
    # 将列表转换为填充后的张量
    all_text_tokens = torch.nn.utils.rnn.pad_sequence(all_text_tokens, batch_first=True, padding_value=0)
    all_speech_tokens = torch.nn.utils.rnn.pad_sequence(all_speech_tokens, batch_first=True, padding_value=0)
    
    # 如果需要填充到最大长度
    if pad_to_max_length and not skip:
        pad_length = max_length - my_max_length
        if pad_length > 0:
            all_speech_tokens = torch.nn.functional.pad(all_speech_tokens, (0, pad_length), value=0)
    
    return {
        'text_token': all_text_tokens, 
        'text_token_len': torch.tensor(text_token_len, dtype=torch.int32), 
        'speech_token': all_speech_tokens,  # 确保命名一致
        'speech_token_len': torch.tensor(speech_token_len, dtype=torch.int32),
        'skip': skip
    }
        
        
if __name__ == '__main__':
    from transformers import AutoTokenizer
    model_path = "/external_data/models/rwkv7-2.9B-world"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    directory = '/external_data/yueyudata/speech_corpus'
    dataset = load_jsonl_dataset(directory,tokenizer)
    print(dataset)
    print(dataset[0])
    from functools import partial
    collate_fn = partial(collate_fn,tokenizer=tokenizer,pad_to_max_length=False)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,collate_fn=collate_fn)
    for data in dataloader:
        print(data)
        print(data['speech_token'].shape)
        print(data['text_token'].shape)
        break