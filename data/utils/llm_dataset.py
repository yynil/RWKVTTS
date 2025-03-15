import datasets
import os
import json
import torch
import random
import time
random.seed(time.time())
import logging
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verify_jsonl_files(data_files):
    """检查每个 jsonl 文件的有效性"""
    invalid_files = []
    
    for file_path in tqdm(data_files, desc="验证文件"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        invalid_files.append((file_path, i+1))
                        logging.error(f"文件 {file_path} 在第 {i+1} 行有无效的 JSON")
                        break
        except Exception as e:
            invalid_files.append((file_path, f"读取错误: {str(e)}"))
            logging.error(f"无法读取文件 {file_path}: {str(e)}")
    
    return invalid_files
def load_jsonl_dataset(directory,tokenizer):
    '''
    load jsonl files in a directory recursively
    '''
    data_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jsonl'):
                data_files.append(os.path.join(root, file))
    
    logging.info(f"找到 {len(data_files)} 个 JSONL 文件")
    # 验证文件
    invalid_files = verify_jsonl_files(data_files)
    if invalid_files:
        logging.error(f"发现 {len(invalid_files)} 个无效文件:")
        for file_info in invalid_files:
            if isinstance(file_info[1], int):
                logging.error(f"  - {file_info[0]} (错误在第 {file_info[1]} 行)")
            else:
                logging.error(f"  - {file_info[0]} ({file_info[1]})")
        
        # 移除无效文件
        valid_files = [f for f in data_files if f not in [info[0] for info in invalid_files]]
        logging.info(f"继续处理剩余的 {len(valid_files)} 个有效文件")
        data_files = valid_files
    # 手动收集所有样本，确保特征一致性
    all_samples = []
    
    for file_path in tqdm(data_files, desc="加载数据集"):
        try:
            # 手动解析JSONL文件，避免datasets加载时的类型推断问题
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # 确保所有字段存在且类型一致
                        llm_prompt_speech_token = data.get('llm_prompt_speech_token', [])
                        tts_speech_tokens = data.get('tts_speech_tokens', [])
                        text = str(data.get('text', ""))
                        prompt_text = str(data.get('prompt_text', ""))
                        
                        # 确保列表类型
                        if not isinstance(llm_prompt_speech_token, list):
                            llm_prompt_speech_token = []
                        if not isinstance(tts_speech_tokens, list):
                            tts_speech_tokens = []
                            
                        # 添加处理后的样本
                        all_samples.append({
                            'llm_prompt_speech_token': llm_prompt_speech_token,
                            'tts_speech_tokens': tts_speech_tokens,
                            'text': text,
                            'prompt_text': prompt_text
                        })
                    except json.JSONDecodeError:
                        continue  # 跳过无效的JSON行
                    except Exception as e:
                        logging.error(f"处理样本时出错: {str(e)}")
        except Exception as e:
            logging.error(f"打开文件 {file_path} 时出错: {str(e)}")
    
    if not all_samples:
        raise ValueError("没有成功加载任何样本")
    
    # 创建数据集
    logging.info(f"手动创建数据集，包含 {len(all_samples)} 个样本")
    dataset = datasets.Dataset.from_list(all_samples)
    
    logging.info(f"成功加载 {len(dataset)} 个样本")
    
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
    dataset = dataset.filter(lambda x:len(x['llm_prompt_speech_token']) < 2048 and len(x['tts_speech_tokens']) < 2048
                             and len(tokenizer.encode(x['text'])) < 2048 and len(tokenizer.encode(x['prompt_text'])) < 2048 )
    logging.info(f"过滤后剩余 {len(dataset)} 个样本")
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