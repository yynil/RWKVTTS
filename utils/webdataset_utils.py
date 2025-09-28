#!/usr/bin/env python3
"""
WebDataset 工具函数
提供音频数据加载、处理和批处理的完整 pipeline
"""

import os
import re
import io
import numpy as np
import soundfile as sf
import librosa
import webdataset as wds
import torch
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import DataLoader
from webdataset.handlers import warn_and_continue

def is_text_low_quality(text: str) -> bool:
    """
    检测文本是否低质量
    如果字符串中功能符太多，如\t\n\r之类的，就返回 True
    
    Args:
        text: 输入文本
        
    Returns:
        bool: True 表示低质量文本，False 表示正常文本
    """
    if not text or not isinstance(text, str):
        return True
    
    # 计算控制字符的数量
    control_chars = ['\t', '\n', '\r', '\f', '\v']
    control_count = sum(text.count(char) for char in control_chars)
    
    # 计算文本总长度
    text_length = len(text)
    
    # 如果文本太短，认为是低质量
    if text_length < 3:
        return True
    
    # 如果控制字符占比超过20%，认为是低质量
    if control_count / text_length > 0.2:
        return True
    
    # 如果控制字符绝对数量超过10个，认为是低质量
    if control_count > 10:
        return True
    
    # 检查是否包含过多的连续制表符（可能是数据记录）
    if '\t\t' in text and text.count('\t') > 5:
        return True
    
    # 检查是否包含过多的连续换行符
    if '\n\n' in text and text.count('\n') > 3:
        return True
    
    return False


def detect_language(text: str) -> str:
    """
    检测文本语言
    
    Args:
        text: 输入文本
        
    Returns:
        str: 'chinese' 或 'english'
    """
    # 检查是否包含中文字符
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    if len(chinese_chars) > 0:
        return 'chinese'
    else:
        return 'english'


def get_language_instruction(language: str) -> Tuple[str, str]:
    """
    根据语言获取指令和提示
    
    Args:
        language: 语言类型 ('chinese' 或 'english')
        
    Returns:
        Tuple[str, str]: (instruction, hints)
    """
    if language == 'chinese':
        instruction = "User: 请将以下语音转写为中文。\n"
        hints = "Assistant: "
    else:
        instruction = "User: Convert the audios to English.\n"
        hints = "Assistant: "
    
    return instruction, hints


def process_audio_sample(audio_data: bytes, target_sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    处理音频样本，统一转换为指定格式
    
    Args:
        audio_data: 音频字节数据
        target_sample_rate: 目标采样率，默认16000
        
    Returns:
        Tuple[np.ndarray, int]: (音频数组, 采样率)
    """
    try:
        # 使用 soundfile 读取音频数据
        audio_buffer = io.BytesIO(audio_data)
        audio_array, sample_rate = sf.read(audio_buffer)
        

        if audio_array.ndim == 1:
            audio_array = audio_array.reshape(1, -1)
        
        # 转换为单声道
        if audio_array.shape[0] > 1:
            audio_array = np.mean(audio_array, axis=0, keepdims=True)
        
        # 重采样到目标采样率
        if sample_rate != target_sample_rate:
            # 使用librosa进行高质量重采样
            # 此时audio_array已经是单声道，直接重采样即可
            audio_array = librosa.resample(
                audio_array[0], 
                orig_sr=sample_rate, 
                target_sr=target_sample_rate
            ).reshape(1, -1)
            sample_rate = target_sample_rate
        
        # 限制音频长度（30秒）
        max_length = target_sample_rate * 30
        if audio_array.shape[1] > max_length:
            audio_array = audio_array[:, :max_length]
        
        return audio_array, sample_rate
        
    except Exception as e:
        print(f"处理音频失败: {e}")
        return None, None


def extract_text_label(sample: Dict[str, Any]) -> str:
    """
    从样本中提取文本标签
    
    Args:
        sample: WebDataset 样本
        
    Returns:
        str: 文本标签
    """
    text_label = ""
    
    # 尝试从 JSON 中提取文本
    if 'json' in sample:
        try:
            import json
            json_data = json.loads(sample['json'])
            text_label = (
                json_data.get('text', '') or 
                json_data.get('transcript', '') or 
                json_data.get('caption', '') or
                json_data.get('label', '')
            )
        except:
            pass
    
    # 如果没有找到文本标签，使用文件名作为标签
    if not text_label:
        # 尝试从音频文件名推断
        for key in ['wav', 'mp3', 'flac']:
            if key in sample:
                # 这里可以根据实际的文件命名规则来提取文本
                text_label = f"audio_{key}"
                break
    
    return text_label if text_label else "unknown"


def process_webdataset_sample(sample: Dict[str, Any], target_sample_rate: int = 16000) -> Optional[Dict[str, Any]]:
    """
    处理 WebDataset 单个样本，具有robust的错误处理
    
    Args:
        sample: WebDataset 样本
        target_sample_rate: 目标采样率
        
    Returns:
        Optional[Dict]: 处理后的样本，如果失败返回 None
    """
    try:
        # 检查样本是否有效
        if not sample or not isinstance(sample, dict):
            return None
            
        # 查找音频文件
        audio_data = None
        audio_format = None
        
        for format_name in ['wav', 'mp3', 'flac']:
            if format_name in sample and sample[format_name] is not None:
                audio_data = sample[format_name]
                audio_format = format_name
                break
        
        if audio_data is None:
            return None
        
        # 处理音频
        audio_array, sample_rate = process_audio_sample(audio_data, target_sample_rate)
        
        # 检查音频处理结果
        if audio_array is None or sample_rate is None:
            return None
            
        # 提取文本标签
        text_label = extract_text_label(sample)
        
        # 检查文本是否有效
        if not text_label or text_label == "unknown":
            return None
        
        # 检测语言
        language = detect_language(text_label)
        
        return {
            'wav': audio_array,  # 返回 numpy 数组，不是 tensor
            'text': text_label,
            'format': audio_format,
            'language': language,
            'sample_rate': sample_rate
        }
        
    except Exception as e:
        print(f"处理样本失败: {e}")
        return None


def create_webdataset_pipeline(
    data_files: List[str],
    world_size: int,
    global_rank: int,
    batch_size: int,
    target_sample_rate: int = 16000,
    num_workers: int = 4,
    shardshuffle: int = 100
) -> Tuple[Any, DataLoader]:
    """
    创建WebDataset数据管道
    
    Args:
        data_files: 数据文件列表
        world_size: 总进程数
        global_rank: 当前进程rank
        batch_size: 批次大小
        target_sample_rate: 目标采样率
        num_workers: 工作进程数
        shardshuffle: shard级别的shuffle大小
        
    Returns:
        dataset: WebDataset对象
        dataloader: DataLoader对象
    """
    print(f"创建WebDataset管道: {data_files} 个文件, world_size={world_size}, global_rank={global_rank}")

    
    # 创建WebDataset，添加错误处理机制
    dataset = wds.DataPipeline(
        wds.SimpleShardList(data_files,seed=True),
        wds.resampled,  # 重新添加resampled，确保数据无限循环
        wds.shuffle(100000),
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=warn_and_continue),
        wds.map(lambda x: process_webdataset_sample(x, target_sample_rate), handler=warn_and_continue),  # 添加错误处理
        wds.select(lambda x: x is not None and x['wav'] is not None and x['text'] is not None),
        wds.select(lambda x: x['wav'].shape[1] != target_sample_rate * 30),
        wds.select(lambda x: not is_text_low_quality(x['text'])),
        wds.to_tuple("wav", "text", "format", "language", "sample_rate"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
        shuffle=False
    )
    return dataset, dataloader


def custom_collate_fn(batch):
    """
    自定义的collate函数，处理不同长度的音频数据
    
    Args:
        batch: 从WebDataset返回的batch
        
    Returns:
        tuple: (wavs, texts, formats, languages, sample_rates)
    """
    # 解包batch
    wavs, texts, formats, languages, sample_rates = zip(*batch)
    
    # 音频数据保持为numpy数组列表，不转换为张量
    # 文本数据保持为字符串列表
    # 格式、语言、采样率保持为列表
    
    return wavs, texts, formats, languages, sample_rates


def process_batch(
    batch: Tuple,
    tokenizer: Any,
    eos_token_id: int = 0,
    pad_token_id: int = 0,
) -> Dict[str, Any]:
    """
    处理 batch，根据语言选择不同的指令
    
    Args:
        batch: 从 DataLoader 返回的 batch
        tokenizer: tokenizer 对象
        eos_token_id: EOS token的ID，默认为0
        pad_token_id: PAD token的ID，默认为0
        
    Returns:
        Dict: 处理后的 batch，包含 instruction, hints, text 分别的 tokens
    """
    # 解包 batch
    wavs, texts, formats, languages, sample_rates = batch
    
    # 处理文本和创建提示
    processed_texts = []
    instruction_tokens_list = []
    hints_tokens_list = []
    text_tokens_list = []
    detected_languages = []
    attention_masks_list = []
    for text, language in zip(texts, languages):
        # 获取语言指令
        instruction, hints = get_language_instruction(language)
        
        # 分别 tokenize instruction, hints, text
        instruction_tokens = tokenizer.encode(instruction)
        hints_tokens = tokenizer.encode(hints)
        text_tokens = tokenizer.encode(text)
        text_tokens = text_tokens + [eos_token_id]
        processed_texts.append(text)
        instruction_tokens_list.append(instruction_tokens)
        hints_tokens_list.append(hints_tokens)
        text_tokens_list.append(text_tokens)
        detected_languages.append(language)
        attention_masks_list.append(torch.ones(len(instruction_tokens), dtype=torch.long))
    # 处理 token 的填充 - 使用 pad_sequence 左对齐
    from torch.nn.utils.rnn import pad_sequence
    
    # 转换为 torch tensor 列表
    instruction_tensors = [torch.tensor(tokens, dtype=torch.long) for tokens in instruction_tokens_list]
    hints_tensors = [torch.tensor(tokens, dtype=torch.long) for tokens in hints_tokens_list]
    text_tokens_tensors = [torch.tensor(tokens, dtype=torch.long) for tokens in text_tokens_list]

    # 使用 pad_sequence 左对齐
    instruction_tokens_batch = pad_sequence(instruction_tensors, batch_first=True, padding_value=pad_token_id,padding_side="left")
    hints_tokens_batch = pad_sequence(hints_tensors, batch_first=True, padding_value=pad_token_id,padding_side="left")
    text_tokens_batch = pad_sequence(text_tokens_tensors, batch_first=True, padding_value=-100,padding_side="left")
    text_attention_mask = pad_sequence(attention_masks_list, batch_first=True, padding_value=0,padding_side="left")
    
    # 创建最终的 batch
    processed_batch = {
        'wavs': wavs,  # 原始 numpy 数组列表
        'texts': processed_texts,
        'formats': formats,
        'languages': detected_languages,
        'sample_rate': sample_rates[0],  # 所有样本都是相同的采样率
        'instruction_tokens': instruction_tokens_batch,  # 分开的 instruction tokens
        'hints_tokens': hints_tokens_batch,             # 分开的 hints tokens
        'text_tokens': instruction_tokens_batch,               # 文本 tokens
        'text_attention_mask': text_attention_mask,     # 文本 attention mask
        'labels': text_tokens_batch.clone()             # labels 是文本 tokens 的副本
    }
    
    
    return processed_batch


def create_complete_pipeline(
    data_files: List[str],
    world_size: int,
    global_rank: int,
    batch_size: int = 4,
    target_sample_rate: int = 16000,
    num_workers: int = 2,
    shardshuffle: int = 100
) -> Tuple[wds.WebDataset, DataLoader]:
    """
    创建完整的 pipeline
    
    Args:
        data_files: 数据文件列表
        world_size: 总进程数
        global_rank: 当前进程编号
        batch_size: 批次大小
        target_sample_rate: 目标采样率
        num_workers: 工作进程数
        shardshuffle: 分片随机化参数
        
    Returns:
        Tuple[wds.WebDataset, DataLoader]: (数据集, 数据加载器)
    """
    print(f"=== 创建完整的 WebDataset Pipeline ===")
    print(f"总文件数: {len(data_files)}")
    print(f"进程数: {world_size}, 当前进程: {global_rank}")
    print(f"批次大小: {batch_size}, 目标采样率: {target_sample_rate}Hz")
    
    # 直接使用 create_webdataset_pipeline，它已经返回了 (dataset, dataloader)
    dataset,dataloader = create_webdataset_pipeline(
        data_files, 
        world_size, 
        global_rank, 
        batch_size,
        target_sample_rate,
        num_workers,
        shardshuffle
    )
    
    print(f"✅ Pipeline 创建完成！")
    
    return dataset, dataloader

