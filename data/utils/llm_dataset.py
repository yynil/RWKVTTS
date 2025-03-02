import datasets
import os
import json
import torch
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
    dataset = dataset.map(lambda x: {'speech_token': x['llm_prompt_speech_token'] + x['tts_speech_tokens']},remove_columns=['tts_speech_tokens','llm_prompt_speech_token'])
    #2. tokenize the text to text_tokens and prompt_text to prompt_text_tokens
    # dataset = dataset.map(lambda x: {'text_tokens': tokenizer.encode(x['text']), 'prompt_text_tokens': tokenizer.encode(x['prompt_text'])},remove_columns=['text','prompt_text'])
    return dataset

def collate_fn(batch,tokenizer):
    '''
    convert the data to torch tensors
    1. call tokenizer.encode('text') and tokenizer.encode('prompt_text'), concatenate them to get the text_token, record each sample's length to text_token_len
    2. convert the text_tokens and text_token_len to torch tensor
    3. record each sample's speech_token length to speech_token_len
    4. convert the speech_token and speech_token_len to torch tensor
    '''
    all_text_tokens = []
    prompt_text_tokens = []
    speech_tokens = []
    speech_token_len = []
    text_token_len = []
    for sample in batch:
        text_tokens = tokenizer.encode(sample['text'])
        prompt_text_tokens = tokenizer.encode(sample['prompt_text'])
        all_tokens = text_tokens + prompt_text_tokens
        all_text_tokens.append(torch.tensor(all_tokens,dtype=torch.int32))
        text_token_len.append(len(all_tokens))
        speech_tokens.append(torch.tensor(sample['speech_token'],dtype=torch.int32))
        speech_token_len.append(len(sample['speech_token']))
    all_text_tokens = torch.nn.utils.rnn.pad_sequence(all_text_tokens,batch_first=True,padding_value=0)
    speech_tokens = torch.nn.utils.rnn.pad_sequence(speech_tokens,batch_first=True,padding_value=0)
    return {'text_token': all_text_tokens, 'text_token_len': torch.tensor(text_token_len,dtype=torch.int32), 'speech_token': speech_tokens, 'speech_token_len': torch.tensor(speech_token_len,dtype=torch.int32)}
        
        
if __name__ == '__main__':
    from transformers import AutoTokenizer
    model_path = "/external_data/models/rwkv7-2.9B-world"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    directory = '/external_data/yueyudata/speech_corpus'
    dataset = load_jsonl_dataset(directory,tokenizer)
    print(dataset)
    print(dataset[0])
    from functools import partial
    collate_fn = partial(collate_fn,tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=2,collate_fn=collate_fn)
    for data in dataloader:
        print(data)
        print(data['speech_token'].shape)
        print(data['text_token'].shape)
        break