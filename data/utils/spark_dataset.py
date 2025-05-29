from datasets import load_dataset
import os
import torch
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def load_spark_jsonl_dataset(directory):
    dataset = load_dataset('json', data_files=[ os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jsonl')])
    return dataset

def convert_to_tts_format(example):
    """将原始格式转换为TTS格式"""
    text = example['text']
    global_tokens = example['global_tokens']
    semantic_tokens = example['semantic_tokens']
    
    # 转换global tokens，每个token单独转换
    global_str = ''.join([f'<|bicodec_global_{token}|>' for token in global_tokens])
    
    # 转换semantic tokens，每个token单独转换
    semantic_str = ''.join([f'<|bicodec_semantic_{token}|>' for token in semantic_tokens])
    
    # 组合最终格式
    formatted_text = f"<tts><text_start>{text}<text_end><global_start>{global_str}<global_end><sementic_start>{semantic_str}<sementic_end>"
    
    return {"text": formatted_text}

def collate_fn(batch, tokenizer, pad_to_max_length=True, max_length=2048, drop_prompt_audio_rate=-0.1,semantic_start_id=65536):
    # 获取所有样本的input_ids
    input_ids_list = []
    for sample in batch:
        input_ids = tokenizer.encode(sample['text'])
        input_ids_list.append(input_ids)
    
    # 找到最长的序列长度
    max_seq_length = max(len(ids) for ids in input_ids_list)
    if pad_to_max_length:
        max_seq_length = max(max_seq_length, max_length)
    
    # 初始化batch的tensors
    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_seq_length), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
    labels = torch.full((batch_size, max_seq_length), -100, dtype=torch.long)
    
    # 填充每个样本
    for i, ids in enumerate(input_ids_list):
        # 如果超过max_length，截断
        if len(ids) > max_seq_length:
            ids = ids[-max_seq_length:]
        
        # 计算padding长度
        pad_length = max_seq_length - len(ids)
        
        # 填充input_ids（左padding）
        input_ids[i, pad_length:] = torch.tensor(ids)
        
        # 设置attention_mask
        attention_mask[i, pad_length:] = 1
        
        # 找到semantic_start的位置
        semantic_start_pos = None
        for j, token_id in enumerate(ids):
            if token_id == semantic_start_id:
                semantic_start_pos = j
                break
        
        # 设置labels
        if semantic_start_pos is not None:
            # semantic_start之前（包括）的位置设为-100
            labels[i, pad_length:pad_length+semantic_start_pos+1] = -100
            # 之后的位置左移一位（注意：最后一个token会被丢弃，因为它是预测目标）
            labels[i, pad_length+semantic_start_pos+1:pad_length+len(ids)-1] = torch.tensor(ids[semantic_start_pos+2:])
        else:
            # 如果没有找到semantic_start，整个序列左移一位
            labels[i, pad_length:pad_length+len(ids)-1] = torch.tensor(ids[1:])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

if __name__ == "__main__":
    dataset = load_spark_jsonl_dataset("/home/yueyulin/data/Emilia/ZH/tar_tokens/")
    print(dataset['train'][0])
    # 使用map转换整个数据集，启用多进程
    converted_dataset = dataset.map(
        convert_to_tts_format,
        num_proc=4,  # 使用4个进程
        remove_columns=dataset['train'].column_names,  # 删除所有原有特征
        desc="Converting to TTS format"  # 显示进度条描述
    )
    print("转换后的数据集示例：")
    print(converted_dataset['train'][0])

    rwkv_model_dir = '/home/yueyulin/models/rwkv7-0.1B-g1'
    from transformers import AutoTokenizer
    from utils.utilities import get_respark_tts_tokenizer
    tokenizer = get_respark_tts_tokenizer(rwkv_model_dir)
    
    # 统计长度分布
    lengths = []
    for sample in converted_dataset['train']:
        input_ids = tokenizer.encode(sample['text'])
        lengths.append(len(input_ids))
    
    # 计算基本统计信息
    lengths = np.array(lengths)
    print(f"\n长度统计信息：")
    print(f"最小长度: {lengths.min()}")
    print(f"最大长度: {lengths.max()}")
    print(f"平均长度: {lengths.mean():.2f}")
    print(f"中位数长度: {np.median(lengths):.2f}")
    print(f"标准差: {lengths.std():.2f}")
    
    # 计算分位数
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        print(f"{p}分位数: {np.percentile(lengths, p):.2f}")
    
    # 绘制长度分布直方图
    plt.figure(figsize=(12, 6))
    plt.hist(lengths, bins=50, alpha=0.7)
    plt.title('Text Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Number of Samples')
    plt.grid(True, alpha=0.3)
    plt.savefig('length_distribution.png')
    plt.close()
    
    # 统计不同长度区间的样本数量
    bins = [0, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, float('inf')]
    hist, _ = np.histogram(lengths, bins=bins)
    print("\n长度区间分布：")
    for i in range(len(bins)-1):
        print(f"{bins[i]}-{bins[i+1]}: {hist[i]} 样本")

    tokenizer = get_respark_tts_tokenizer(rwkv_model_dir)
    # input_ids = tokenizer.encode(converted_dataset['train'][0]['text'])
    # print(input_ids)
    # text_start_id = tokenizer.encode("<text_start>")[0]
    # text_end_id = tokenizer.encode("<text_end>")[0]
    # global_start_id = tokenizer.encode("<global_start>")[0]
    # global_end_id = tokenizer.encode("<global_end>")[0]
    # semantic_start_id = tokenizer.encode("<sementic_start>")[0]
    # semantic_end_id = tokenizer.encode("<sementic_end>")[0]

    # global_start_index = input_ids.index(global_start_id)
    # global_end_index = input_ids.index(global_end_id)   
    # semantic_start_index = input_ids.index(semantic_start_id)
    # semantic_end_index = input_ids.index(semantic_end_id)
    # text_start_index = input_ids.index(text_start_id)
    # text_end_index = input_ids.index(text_end_id)

    # global_tokens = input_ids[global_start_index+1:global_end_index]
    # semantic_tokens = input_ids[semantic_start_index+1:semantic_end_index]
    # text = input_ids[text_start_index+1:text_end_index]

    # print(global_tokens)
    # print(semantic_tokens)
    # print(text)

    # print(f'global_start_index: {global_start_index}, global_end_index: {global_end_index}, semantic_start_index: {semantic_start_index}, semantic_end_index: {semantic_end_index}, global_start_id: {global_start_id}, global_end_id: {global_end_id}, semantic_start_id: {semantic_start_id}, semantic_end_id: {semantic_end_id}')
    # print(f'{tokenizer.decode(text)}')
    # global_tokens = [int(tokenizer.decode(g).replace('<|bicodec_global_', '').replace('|>', '')) for g in global_tokens]
    # semantic_tokens = [int(tokenizer.decode(s).replace('<|bicodec_semantic_', '').replace('|>', '')) for s in semantic_tokens]
    # print(global_tokens)
    # print(semantic_tokens)

    from torch.utils.data import DataLoader
    from functools import partial
    dataloader = DataLoader(converted_dataset['train'], batch_size=1, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=4)
    for batch in dataloader:
        print(batch)
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        print(input_ids)
        print(attention_mask)
        print(labels)
        print(input_ids.shape, attention_mask.shape, labels.shape)
        break

    