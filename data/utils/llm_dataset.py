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
    #2. Filter the data either :
    #   1. the length of the speech_token is less than 1
    #   2. the length of the speech_token is greater than 1000
    #   3. the length of the text is greater than 500
    #   4. the length of the prompt_text is greater than 500
    #   5. the length of the text_token is less than 1
    #   6. the length of the prompt_text_token is less than 1
    dataset = dataset.filter(lambda x: len(x['speech_token']) > 1 and len(x['speech_token']) < 1000 and len(tokenizer.encode(x['text'])) < 500 and len(tokenizer.encode(x['prompt_text'])) < 500 and len(tokenizer.encode(x['text'])) > 1 and len(tokenizer.encode(x['prompt_text'])) > 1)
    #2. tokenize the text to text_tokens and prompt_text to prompt_text_tokens
    # dataset = dataset.map(lambda x: {'text_tokens': tokenizer.encode(x['text']), 'prompt_text_tokens': tokenizer.encode(x['prompt_text'])},remove_columns=['text','prompt_text'])
    return dataset

def collate_fn(batch,tokenizer,pad_to_max_length=True,max_length=2048):
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
    my_max_length = 0
    for sample in batch:
        text_tokens = tokenizer.encode(sample['text'])
        prompt_text_tokens = tokenizer.encode(sample['prompt_text'])
        all_tokens = prompt_text_tokens + text_tokens
        all_text_tokens.append(torch.tensor(all_tokens,dtype=torch.int32))
        text_token_len.append(len(all_tokens))
        speech_tokens.append(torch.tensor(sample['speech_token'],dtype=torch.int32))
        speech_token_len.append(len(sample['speech_token']))
        total_length = len(all_tokens) + len(sample['speech_token'])
        if total_length > my_max_length:
            my_max_length = total_length
    if my_max_length > max_length:
        skip = True#skip this sample
    else:
        skip = False
    
    all_text_tokens = torch.nn.utils.rnn.pad_sequence(all_text_tokens,batch_first=True,padding_value=0)
    speech_tokens = torch.nn.utils.rnn.pad_sequence(speech_tokens,batch_first=True,padding_value=0)
    #pad pad_length to the speech_tokens
    if pad_to_max_length:
        pad_length =max_length - my_max_length
        if pad_length > 0:
            speech_tokens = torch.nn.functional.pad(speech_tokens,(0,pad_length),value=0)
    return {'text_token': all_text_tokens, 'text_token_len': torch.tensor(text_token_len,dtype=torch.int32), 'speech_token': speech_tokens, 'speech_token_len': torch.tensor(speech_token_len,dtype=torch.int32),'skip':skip}
        
        
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