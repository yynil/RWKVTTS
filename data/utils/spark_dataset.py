from datasets import load_dataset
import os
import torch
from collections import Counter
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.rwkv7speech_inference import create_inputs

def load_spark_jsonl_dataset(directory):
    # 递归获取所有jsonl文件
    jsonl_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_files.append(os.path.join(root, file))
    
    # 加载所有找到的jsonl文件
    dataset = load_dataset('json', data_files=jsonl_files)
    print(f'load {len(jsonl_files)} jsonl files')
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

def collate_fn_for_rwkv7speech(batch,tokenizer,rwkv7speech_model,max_length=2048,pad_to_max_length=True,vocab_size=8193):
    
    device = rwkv7speech_model.device
    texts = [sample['text'] for sample in batch]
    global_tokens_ids = [sample['global_tokens'] for sample in batch]
    semantic_tokens_ids = [sample['semantic_tokens']+[vocab_size-1] for sample in batch]
    input_ids_embs,attention_mask = create_inputs(texts,global_tokens_ids,semantic_tokens_ids,tokenizer,rwkv7speech_model)  
    labels = torch.full((input_ids_embs.shape[0],input_ids_embs.shape[1]),-100,dtype=torch.long,device=device)
    for i in range(len(semantic_tokens_ids)):
        labels[i,-(len(semantic_tokens_ids[i])+1):-1] = torch.tensor(semantic_tokens_ids[i], device=device)
    return {"input_embs": input_ids_embs,"attention_mask": attention_mask,"labels": labels}



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

def process_single_batch_culens(batch, rwkv7speech_model,eos_token_id=8192,max_cu_seqlens=8192):
    """
    处理单个batch，生成与collate_fn_for_rwkv7speech相同格式的输出
    
    Args:
        batch: 包含input_ids, attention_mask_input_ids, global_tokens_ids, global_tokens_attention_mask,
              semantic_tokens_ids, semantic_tokens_attention_mask的字典
        rwkv7speech_model: 模型
    
    Returns:
        包含input_embs, labels,cu_seqlens的字典
    """
    device = rwkv7speech_model.device
    batch_size = batch['input_ids'].shape[0]
    input_ids_embs_list = []
    labels = []
    cu_seqlens = [0]
    for i in range(batch_size):
        text_length = batch['attention_mask_input_ids'][i].sum().item()
        input_ids = batch['input_ids'][i, -text_length:]
        global_length = batch['global_tokens_attention_mask'][i].sum().item()
        global_ids = batch['global_tokens_ids'][i, -global_length:]
        semantic_length = batch['semantic_tokens_attention_mask'][i].sum().item()
        semantic_ids = batch['semantic_tokens_ids'][i, -semantic_length:]

        # 获取embeddings
        input_ids_embs = rwkv7speech_model.text_embedder(input_ids.to(device))#T,D
        global_tokens_embs = rwkv7speech_model.global_embedder(global_ids.to(device))#G_T,D
        semantic_tokens_embs = rwkv7speech_model.model.embeddings(semantic_ids.to(device))#S_T,D
        
        # 获取tts tag embeddings
        tts_tag_embedder_0 = rwkv7speech_model.tts_tag_embedder(torch.tensor([0], dtype=torch.long, device=device))#1,D
        tts_tag_embedder_1 = rwkv7speech_model.tts_tag_embedder(torch.tensor([1], dtype=torch.long, device=device))#1,D
        tts_tag_embedder_2 = rwkv7speech_model.tts_tag_embedder(torch.tensor([2], dtype=torch.long, device=device))#1,D
        
        # 拼接embeddings
        input_ids_embs = torch.cat([tts_tag_embedder_2,input_ids_embs, tts_tag_embedder_0, global_tokens_embs, tts_tag_embedder_1, semantic_tokens_embs], dim=0)#T+1+G_T+1+S_T,D
        input_ids_embs_list.append(input_ids_embs)
        my_length = input_ids_embs.shape[0]
        my_label = torch.full((my_length,),-100,dtype=torch.long,device=device)
        my_label[-semantic_length-1:-1] = semantic_ids
        my_label[-1] = eos_token_id
        labels.append(my_label)
        last_length = cu_seqlens[-1]
        if last_length+my_length > max_cu_seqlens:
            break
        cu_seqlens.append(last_length+my_length)
    input_embs = torch.cat(input_ids_embs_list,dim=0)
    labels = torch.cat(labels,dim=0)
    cu_seqlens = torch.tensor(cu_seqlens,dtype=torch.long,device=device)

    return {"input_embs": input_embs.unsqueeze(0),"labels": labels.unsqueeze(0),"cu_seqlens": cu_seqlens}
def process_single_batch(batch, rwkv7speech_model,eos_token_id=8192):
    """
    处理单个batch，生成与collate_fn_for_rwkv7speech相同格式的输出
    
    Args:
        batch: 包含input_ids, attention_mask_input_ids, global_tokens_ids, global_tokens_attention_mask,
              semantic_tokens_ids, semantic_tokens_attention_mask的字典
        rwkv7speech_model: 模型
    
    Returns:
        包含input_embs, attention_mask, labels的字典
    """
    device = rwkv7speech_model.device
    batch_size = batch['input_ids'].shape[0]
    
    # 获取每个样本的实际数据（去除填充）
    input_ids_embs_list = []
    max_length = 0
    
    for i in range(batch_size):
        # 获取文本
        text_length = batch['attention_mask_input_ids'][i].sum().item()
        input_ids = batch['input_ids'][i, -text_length:]
        
        # 获取全局标记
        global_length = batch['global_tokens_attention_mask'][i].sum().item()
        global_ids = batch['global_tokens_ids'][i, -global_length:]
        
        # 获取语义标记
        semantic_length = batch['semantic_tokens_attention_mask'][i].sum().item()
        semantic_ids = batch['semantic_tokens_ids'][i, -semantic_length:]
        
        # 获取embeddings
        input_ids_embs = rwkv7speech_model.text_embedder(input_ids.unsqueeze(0).to(device))
        global_tokens_embs = rwkv7speech_model.global_embedder(global_ids.unsqueeze(0).to(device))
        semantic_tokens_embs = rwkv7speech_model.model.embeddings(semantic_ids.unsqueeze(0).to(device))
        
        # 获取tts tag embeddings
        tts_tag_embedder_0 = rwkv7speech_model.tts_tag_embedder(torch.tensor([[0]], dtype=torch.long, device=device))
        tts_tag_embedder_1 = rwkv7speech_model.tts_tag_embedder(torch.tensor([[1]], dtype=torch.long, device=device))
        tts_tag_embedder_2 = rwkv7speech_model.tts_tag_embedder(torch.tensor([[2]], dtype=torch.long, device=device))
        # 拼接embeddings
        input_ids_embs = torch.cat([tts_tag_embedder_2,input_ids_embs, tts_tag_embedder_0, global_tokens_embs, tts_tag_embedder_1, semantic_tokens_embs], dim=1)
        input_ids_embs_list.append(input_ids_embs)
        max_length = max(max_length, input_ids_embs.shape[1])
    
    # 左填充和创建attention mask
    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
    for i in range(batch_size):
        attention_mask[i, -input_ids_embs_list[i].shape[1]:] = 1
        embs_pad_t = max_length - input_ids_embs_list[i].shape[1]
        # 左填充input_ids_embs_list[i]
        input_ids_embs_list[i] = torch.cat([
            torch.zeros(1, embs_pad_t, input_ids_embs_list[i].shape[2], dtype=torch.long, device=device),
            input_ids_embs_list[i]
        ], dim=1)
    
    input_embs = torch.cat(input_ids_embs_list, dim=0)
    
    # 设置labels
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long, device=device)
    for i in range(batch_size):
        semantic_length = batch['semantic_tokens_attention_mask'][i].sum().item()
        semantic_ids = batch['semantic_tokens_ids'][i, -semantic_length:]
        # 找到semantic tokens在attention mask中的起始位置
        semantic_start = (attention_mask[i] == 1).nonzero()[-semantic_length].item()
        
        # 设置labels：从tts_tag_embedder_1的位置开始预测
        # 即从semantic_start - 1的位置开始设置labels
        labels[i, -semantic_length-1:-1] = semantic_ids
        labels[i,-1]=eos_token_id
    
    return {
        "input_embs": input_embs,
        "attention_mask": attention_mask,
        "labels": labels
    }

def collate_fn_simple(batch, tokenizer, pad_to_max_length=True, max_length=2048):
    """
    简单的数据整理函数，处理文本、全局标记和语义标记的对齐，使用左对齐方式（左边填充）
    所有填充值统一使用0，由attention mask决定有效位置
    
    Args:
        batch: 批次数据
        tokenizer: 分词器
        pad_to_max_length: 是否填充到最大长度
        max_length: 最大序列长度
    
    Returns:
        包含对齐后的各种标记和注意力掩码的字典
    """
    # 获取所有样本的input_ids
    input_ids_list = []
    global_tokens_list = []
    semantic_tokens_list = []
    
    for sample in batch:
        input_ids = tokenizer.encode(sample['text'])
        input_ids_list.append(input_ids)
        global_tokens_list.append(sample['global_tokens'])
        semantic_tokens_list.append(sample['semantic_tokens'])
    
    # 找到最长的序列长度
    max_text_length = max(len(ids) for ids in input_ids_list)
    max_global_length = max(len(tokens) for tokens in global_tokens_list)
    max_semantic_length = max(len(tokens) for tokens in semantic_tokens_list)
    
    if pad_to_max_length:
        max_text_length = min(max_text_length, max_length)
        max_global_length = min(max_global_length, max_length)
        max_semantic_length = min(max_semantic_length, max_length)
    
    # 初始化batch的tensors
    batch_size = len(batch)
    
    # 文本相关的tensors
    input_ids = torch.full((batch_size, max_text_length), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask_input_ids = torch.zeros((batch_size, max_text_length), dtype=torch.long)
    
    # 全局标记相关的tensors
    global_tokens_ids = torch.zeros((batch_size, max_global_length), dtype=torch.long)
    global_tokens_attention_mask = torch.zeros((batch_size, max_global_length), dtype=torch.long)
    
    # 语义标记相关的tensors
    semantic_tokens_ids = torch.zeros((batch_size, max_semantic_length), dtype=torch.long)
    semantic_tokens_attention_mask = torch.zeros((batch_size, max_semantic_length), dtype=torch.long)
    
    # 填充每个样本
    for i in range(batch_size):
        # 处理文本
        text_ids = input_ids_list[i]
        if len(text_ids) > max_text_length:
            text_ids = text_ids[:max_text_length]  # 从开头截断
        pad_length = max_text_length - len(text_ids)
        input_ids[i, pad_length:] = torch.tensor(text_ids)
        attention_mask_input_ids[i, pad_length:] = 1
        
        # 处理全局标记
        global_tokens = global_tokens_list[i]
        if len(global_tokens) > max_global_length:
            global_tokens = global_tokens[:max_global_length]  # 从开头截断
        pad_length = max_global_length - len(global_tokens)
        global_tokens_ids[i, pad_length:] = torch.tensor(global_tokens)
        global_tokens_attention_mask[i, pad_length:] = 1
        
        # 处理语义标记
        semantic_tokens = semantic_tokens_list[i]
        if len(semantic_tokens) > max_semantic_length:
            semantic_tokens = semantic_tokens[:max_semantic_length]  # 从开头截断
        pad_length = max_semantic_length - len(semantic_tokens)
        semantic_tokens_ids[i, pad_length:] = torch.tensor(semantic_tokens)
        semantic_tokens_attention_mask[i, pad_length:] = 1
    
    return {
        "input_ids": input_ids,
        "attention_mask_input_ids": attention_mask_input_ids,
        "global_tokens_ids": global_tokens_ids,
        "global_tokens_attention_mask": global_tokens_attention_mask,
        "semantic_tokens_ids": semantic_tokens_ids,
        "semantic_tokens_attention_mask": semantic_tokens_attention_mask
    }

if __name__ == "__main__":
    dataset = load_spark_jsonl_dataset("/home/yueyulin/data/Emilia/partitioned/")['train']
    print(dataset)
    model_dir = '/home/yueyulin/models/rwkv7-0.1B-g1-respark-speech/'
    model_dir = '/home/yueyulin/models/rwkv7-191M-world-respark'
    device = "cuda:3"
    rwkv7speech_model = AutoModelForCausalLM.from_pretrained(model_dir,trust_remote_code=True).bfloat16().to(device)
    rwkv7speech_model.train()
    tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
    print(rwkv7speech_model)
    from functools import partial
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    
    # 定义优化器
    optimizer = AdamW(rwkv7speech_model.parameters(), lr=1e-4)
    
    collate_fn_simple = partial(collate_fn_simple,tokenizer=tokenizer)
    dataloader = DataLoader(dataset,batch_size=16,collate_fn=collate_fn_simple,shuffle=True)
    rwkv7speech_model.gradient_checkpointing_enable()
    for batch in dataloader:
        # print(batch)
        # print(batch['semantic_tokens_ids'].tolist())
        print("--------------------------------")
        processed_batch = process_single_batch(batch, rwkv7speech_model)
        # print(processed_batch['labels'].tolist())
        print(f'input_embs shape:{processed_batch["input_embs"].shape}, attention_mask shape:{processed_batch["attention_mask"].shape}, labels shape:{processed_batch["labels"].shape}')
        
        # 清零梯度
        optimizer.zero_grad()
        
        outputs = rwkv7speech_model.forward(inputs_embeds=processed_batch['input_embs'],attention_mask=processed_batch['attention_mask'],labels=processed_batch['labels'],use_cache=False)
        print(outputs)
        outputs.loss.backward()
        
        # 更新参数
        optimizer.step()

        optimizer.zero_grad()
        processed_batch = process_single_batch_culens(batch, rwkv7speech_model,max_cu_seqlens=4096)
        print(processed_batch)
        print(f'input_embs shape:{processed_batch["input_embs"].shape}, cu_seqlens shape:{processed_batch["cu_seqlens"].shape}, labels shape:{processed_batch["labels"].shape}')
        outputs = rwkv7speech_model.forward(inputs_embeds=processed_batch['input_embs'],labels=processed_batch['labels'],use_cache=False,cu_seqlens=processed_batch['cu_seqlens'])
        print(outputs)
        outputs.loss.backward()
        optimizer.step()
        
        break