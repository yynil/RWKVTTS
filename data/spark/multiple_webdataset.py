import os
import json
import torch
import torchaudio
import numpy as np
from datasets import load_dataset, concatenate_datasets, Audio
from typing import Dict, List, Optional, Union
from pathlib import Path
import tarfile
import logging
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.data import DataLoader
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from model.llm.spark_llm import RWKV7ForSpeech
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultipleWebDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str = None,
        data_files: List[str] = None,
        target_sr: int = 16000,
        target_channels: int = 1,
        shuffle: bool = True,
        verify_tar: bool = False
    ):
        """
        初始化多数据集管理器
        
        Args:
            data_dir: 包含多个数据集子目录的根目录
            target_sr: 目标采样率
            target_channels: 目标声道数
            shuffle: 是否打乱数据
            verify_tar: 是否验证 tar 文件完整性
        """
        
        self.target_sr = target_sr
        self.target_channels = target_channels
        self.verify_tar = verify_tar
        
        # 为每个子数据集创建 Dataset
        self.datasets = []
        if data_files is not None and len(data_files) > 0:
            for data_file in data_files:
                dataset = load_dataset("webdataset", data_files=data_file,split="train")
                audio = Audio(sampling_rate=target_sr, mono=(target_channels==1))
                features = dataset.features
                audio_key = None
                for key in features.keys():
                    if isinstance(features[key], Audio):
                        audio_key = key
                        break
                if audio_key is None:
                    raise ValueError(f"在数据集中未找到音频数据，可用的键: {list(features.keys())}")
                
                dataset = dataset.cast_column(audio_key, audio)
                #rename audio_key to audio 
                if audio_key != "audio":
                    dataset = dataset.rename_column(audio_key, "audio")
                logger.info(f"成功加载数据集: {data_file}")
                self.datasets.append(dataset)
        else:   
            # 获取所有子数据集目录
            self.data_dir = Path(data_dir)
            if self.data_dir.is_file():
                print(f'load single tar file: {self.data_dir}')
                dataset = load_dataset("webdataset", data_files=str(self.data_dir),split="train")
                audio = Audio(sampling_rate=target_sr, mono=(target_channels==1))
                features = dataset.features
                audio_key = None
                for key in features.keys():
                    if isinstance(features[key], Audio):
                        audio_key = key
                        break
                if audio_key is None:
                    raise ValueError(f"在数据集中未找到音频数据，可用的键: {list(features.keys())}")
                
                dataset = dataset.cast_column(audio_key, audio)
                #rename audio_key to audio 
                if audio_key != "audio":
                    dataset = dataset.rename_column(audio_key, "audio")
                logger.info(f"成功加载数据集: {self.data_dir}")
                self.datasets.append(dataset)
            else:
                self.sub_datasets = [d for d in self.data_dir.iterdir() if d.is_dir()]
                logger.info(f"找到 {len(self.sub_datasets)} 个子数据集目录")
                
                for sub_dir in self.sub_datasets:
                    try:
                        print(f'验证数据集: {sub_dir}')
                        # 获取该目录下的所有 tar 文件
                        tar_files = []
                        for f in sub_dir.glob("*.tar*"):
                            # 验证 tar 文件是否完整
                            if self._is_valid_tar(f):
                                tar_files.append(str(f))
                            else:
                                logger.warning(f"跳过损坏的 tar 文件: {f}")
                        
                        if not tar_files:
                            logger.warning(f"在 {sub_dir} 中没有找到有效的 tar 文件")
                            continue
                            
                        logger.info(f"在 {sub_dir} 中找到 {len(tar_files)} 个有效的 tar 文件")
                        
                        # 使用 datasets 加载 WebDataset
                        dataset = load_dataset(
                            "webdataset",
                            data_files=tar_files,
                            split="train"
                        )
                        
                        audio = Audio(sampling_rate=target_sr, mono=(target_channels==1))
                        features = dataset.features
                        audio_key = None
                        for key in features.keys():
                            if isinstance(features[key], Audio):
                                audio_key = key
                                break
                        if audio_key is None:
                            raise ValueError(f"在数据集中未找到音频数据，可用的键: {list(features.keys())}")
                        
                        dataset = dataset.cast_column(audio_key, audio)
                        #rename audio_key to audio 
                        if audio_key != "audio":
                            dataset = dataset.rename_column(audio_key, "audio")
                        logger.info(f"成功加载数据集: {sub_dir.name}")
                        self.datasets.append(dataset)
                        
                    except Exception as e:
                        logger.error(f"加载数据集 {sub_dir} 时出错: {str(e)}")
                        continue
                    
            if not self.datasets:
                raise ValueError("没有成功加载任何数据集")
                
        # 合并所有数据集
        self.dataset = concatenate_datasets(self.datasets)
        
        # 如果需要打乱数据
        if shuffle:
            self.dataset = self.dataset.shuffle()
            
    def _is_valid_tar(self, tar_path: Path) -> bool:
        """
        检查 tar 文件是否完整
        
        Args:
            tar_path: tar 文件路径
            
        Returns:
            文件是否有效
        """
        if not self.verify_tar:
            return True
        try:
            with tarfile.open(tar_path) as tar:
                # 尝试读取文件列表
                tar.getmembers()
            return True
        except Exception as e:
            logger.warning(f"tar 文件 {tar_path} 损坏: {str(e)}")
            return False
        
    def __getitem__(self, idx: int) -> Dict:
        """
        获取指定索引的样本
        
        Args:
            idx: 样本索引
            
        Returns:
            处理后的样本
        """
        try:
            return self.dataset[idx]
        except Exception as e:
            print(f"Error in __getitem__: {e}")
            print(f"idx: {idx}")
            print(f"self.dataset: {self.dataset}")
            raise e
        
    def __len__(self) -> int:
        """
        返回数据集大小
        
        Returns:
            数据集中的样本总数
        """
        return len(self.dataset)
collate_fn_call_count = 0
collate_fn_call_samples_num = 0
callate_fn_call_time = 0
import time
def collate_fn_with_bicodec(batch, tokenizer, bicodec_tokenizer, pad_to_max_length=True, max_length=2048):
    """
    使用 BiCodecTokenizer 处理音频数据的 collate 函数
    
    Args:
        batch: 批次数据
        tokenizer: 文本分词器
        bicodec_tokenizer: BiCodecTokenizer 实例
        pad_to_max_length: 是否填充到最大长度
        max_length: 最大序列长度
    
    Returns:
        包含对齐后的各种标记和注意力掩码的字典
    """
    global collate_fn_call_count, collate_fn_call_samples_num, callate_fn_call_time
    start_time = time.time()
    # 获取所有样本的input_ids
    input_ids_list = []
    global_tokens_list = []
    semantic_tokens_list = []
    input_ids_len = []
    global_tokens_len = []
    semantic_tokens_len = []
    texts = []
    for sample in batch:
        # 处理文本
        try:
            json_data = sample['json']
            to_be_added_texts = json_data['text']
            input_ids = tokenizer.encode(json_data['text'])
            to_be_added_input_ids = torch.tensor(input_ids,dtype=torch.long)
            # 处理音频
            audio_array = sample['audio']['array']
            # 使用 BiCodecTokenizer 处理音频
            with torch.no_grad():
                # 直接传入 numpy 数组
                global_tokens, semantic_tokens = bicodec_tokenizer.tokenize(audio_array)
                global_tokens = global_tokens.squeeze(0).squeeze(0)
                semantic_tokens = semantic_tokens.squeeze(0)
            input_ids_list.append(to_be_added_input_ids)
            input_ids_len.append(len(input_ids))
            global_tokens_list.append(global_tokens)
            semantic_tokens_list.append(semantic_tokens)
            global_tokens_len.append(global_tokens.shape[0])
            semantic_tokens_len.append(semantic_tokens.shape[0])
            texts.append(to_be_added_texts)
        except Exception as e:
            print(f"Error in collate_fn_with_bicodec: {e}")
            print(f"sample: {sample}")
            print(f"skip this sample")
            continue
    from torch.nn.utils.rnn import pad_sequence
    padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id,padding_side='left')
    padded_global_tokens = pad_sequence(global_tokens_list, batch_first=True, padding_value=0,padding_side='left')
    padded_semantic_tokens = pad_sequence(semantic_tokens_list, batch_first=True, padding_value=0,padding_side='left')
    padded_input_ids_len = torch.tensor(input_ids_len,dtype=torch.long)
    padded_global_tokens_len = torch.tensor(global_tokens_len,dtype=torch.long)
    padded_semantic_tokens_len = torch.tensor(semantic_tokens_len,dtype=torch.long)
    end_time = time.time()
    callate_fn_call_time += end_time - start_time
    collate_fn_call_count += 1
    collate_fn_call_samples_num += len(batch)
    if collate_fn_call_count % 100 == 0:
        print(f"collate_fn_call_count: {collate_fn_call_count}, collate_fn_call_samples_num: {collate_fn_call_samples_num}, callate_fn_call_time: {callate_fn_call_time},avg_callate_fn_call_time: {callate_fn_call_time/collate_fn_call_count}")
    return {
        "input_ids": padded_input_ids,
        "global_tokens_ids": padded_global_tokens,
        "semantic_tokens_ids": padded_semantic_tokens,
        "input_ids_len": padded_input_ids_len,
        "global_tokens_len": padded_global_tokens_len,
        "semantic_tokens_len": padded_semantic_tokens_len,
        "texts": texts
    }

def collate_fn_with_tokenizer(batch, tokenizer, pad_to_max_length=True, max_length=2048):
    """
    使用 BiCodecTokenizer 处理音频数据的 collate 函数
    
    Args:
        batch: 批次数据
        tokenizer: 文本分词器
        pad_to_max_length: 是否填充到最大长度
        max_length: 最大序列长度
    
    Returns:
        包含对齐后的各种标记和注意力掩码的字典
    """
    global collate_fn_call_count, collate_fn_call_samples_num, callate_fn_call_time
    start_time = time.time()
    # 获取所有样本的input_ids
    input_ids_list = []
    input_ids_len = []
    audios = []
    texts = []
    for sample in batch:
        # 处理文本
        json_data = sample['json']
        input_ids = tokenizer.encode(json_data['text'])
        input_ids_list.append(torch.tensor(input_ids,dtype=torch.long))
        input_ids_len.append(len(input_ids))
        texts.append(json_data['text'])
        # 处理音频
        audio_array = sample['audio']['array']
        audios.append(audio_array)
    from torch.nn.utils.rnn import pad_sequence
    padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id,padding_side='left')
    padded_input_ids_len = torch.tensor(input_ids_len,dtype=torch.long)
    end_time = time.time()
    callate_fn_call_time += end_time - start_time
    collate_fn_call_count += 1
    collate_fn_call_samples_num += len(batch)
    if collate_fn_call_count % 100 == 0:
        print(f"collate_fn_call_count: {collate_fn_call_count}, collate_fn_call_samples_num: {collate_fn_call_samples_num}, callate_fn_call_time: {callate_fn_call_time},avg_callate_fn_call_time: {callate_fn_call_time/collate_fn_call_count}")
    return {
        "input_ids": padded_input_ids,
        "input_ids_len": padded_input_ids_len,
        "audios": audios,
        "texts": texts
    }
def process_single_batch_with_audio_tokenizer(batch, rwkv7speech_model,audio_tokenizer,eos_token_id=8192):
    """
    batch:{
        "input_ids": padded_input_ids,
        "input_ids_len": padded_input_ids_len,
        "audios": audios,
        "texts": texts
    }
    """
    device = rwkv7speech_model.device
    batch_size = batch['input_ids'].shape[0]
    
    # 获取每个样本的实际数据（去除填充）
    input_ids_embs_list = []
    max_length = 0
    semantic_tokens_len = []
    semantic_tokens_list = []
    for i in range(batch_size):
        input_ids = batch['input_ids'][i]
        audio_array = batch['audios'][i]
        global_tokens, semantic_tokens = audio_tokenizer.tokenize(audio_array)
        global_tokens = global_tokens.squeeze(0)
        semantic_tokens = semantic_tokens
        semantic_tokens_len.append(semantic_tokens.shape[1])
        semantic_tokens_list.append(semantic_tokens.squeeze(0))
        input_ids_embs = rwkv7speech_model.text_embedder(input_ids.unsqueeze(0).to(device))
        global_tokens_embs = rwkv7speech_model.global_embedder(global_tokens.to(device))
        semantic_tokens_embs = rwkv7speech_model.model.embeddings(semantic_tokens.to(device))
        tts_tag_embedder_2 = rwkv7speech_model.tts_tag_embedder(torch.tensor([[2]], dtype=torch.long, device=device))
        tts_tag_embedder_0 = rwkv7speech_model.tts_tag_embedder(torch.tensor([[0]], dtype=torch.long, device=device))
        tts_tag_embedder_1 = rwkv7speech_model.tts_tag_embedder(torch.tensor([[1]], dtype=torch.long, device=device))
        input_ids_embs = torch.cat([tts_tag_embedder_2,input_ids_embs, tts_tag_embedder_0, global_tokens_embs, tts_tag_embedder_1, semantic_tokens_embs], dim=1)
        input_ids_embs_list.append(input_ids_embs)
        max_length = max(max_length, input_ids_embs.shape[1])
    # 左填充和创建attention mask    
    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
    for i in range(batch_size):
        attention_mask[i, -input_ids_embs_list[i].shape[1]:] = 1
        embs_pad_t = max_length - input_ids_embs_list[i].shape[1]
        input_ids_embs_list[i] = torch.cat([
            torch.zeros(1, embs_pad_t, input_ids_embs_list[i].shape[2], dtype=torch.long, device=device),
            input_ids_embs_list[i]
        ], dim=1)
    
    input_embs = torch.cat(input_ids_embs_list, dim=0)
    
    # 设置labels
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long, device=device)
    for i in range(batch_size): 
        semantic_ids = semantic_tokens_list[i]
        semantic_length = semantic_ids.shape[0]
        
        # 设置labels：从tts_tag_embedder_1的位置开始预测
        # 即从semantic_start - 1的位置开始设置labels
        labels[i, -semantic_length-1:-1] = semantic_ids
        labels[i,-1]=eos_token_id
    
    return {
        "input_embs": input_embs,
        "attention_mask": attention_mask,
        "labels": labels
    }

def process_single_batch_with_audio_tokenizer_culens(batch, rwkv7speech_model, audio_tokenizer, eos_token_id=8192, max_cu_seqlens=8192):
    """
    处理单个batch，使用累积序列长度（cu_seqlens）来管理序列长度
    
    Args:
        batch: {
            "input_ids": padded_input_ids,
            "input_ids_len": padded_input_ids_len,
            "audios": audios,
            "texts": texts
        }
        rwkv7speech_model: 模型
        audio_tokenizer: 音频分词器
        eos_token_id: 结束标记ID
        max_cu_seqlens: 最大累积序列长度
    
    Returns:
        包含input_embs, labels, cu_seqlens的字典
    """
    device = rwkv7speech_model.device
    batch_size = batch['input_ids'].shape[0]
    input_ids_embs_list = []
    labels = []
    cu_seqlens = [0]
    
    for i in range(batch_size):
        # 获取文本和音频数据
        input_ids = batch['input_ids'][i]
        audio_array = batch['audios'][i]
        
        # 使用音频分词器处理音频
        global_tokens, semantic_tokens = audio_tokenizer.tokenize(audio_array)
        # 确保维度正确
        if len(global_tokens.shape) == 3:
            global_tokens = global_tokens.squeeze(0).squeeze(0)  # [G]
        elif len(global_tokens.shape) == 2:
            global_tokens = global_tokens.squeeze(0)  # [G]
            
        if len(semantic_tokens.shape) == 3:
            semantic_tokens = semantic_tokens.squeeze(0).squeeze(0)  # [S]
        elif len(semantic_tokens.shape) == 2:
            semantic_tokens = semantic_tokens.squeeze(0)  # [S]
        
        # 获取embeddings
        input_ids_embs = rwkv7speech_model.text_embedder(input_ids.unsqueeze(0).to(device)).squeeze(0)  # [T, D]
        global_tokens_embs = rwkv7speech_model.global_embedder(global_tokens.unsqueeze(0).to(device)).squeeze(0)  # [G, D]
        semantic_tokens_embs = rwkv7speech_model.model.embeddings(semantic_tokens.unsqueeze(0).to(device)).squeeze(0)  # [S, D]
        
        # 获取tts tag embeddings
        tts_tag_embedder_0 = rwkv7speech_model.tts_tag_embedder(torch.tensor([0], dtype=torch.long, device=device))  # [1, D]
        tts_tag_embedder_1 = rwkv7speech_model.tts_tag_embedder(torch.tensor([1], dtype=torch.long, device=device))  # [1, D]
        tts_tag_embedder_2 = rwkv7speech_model.tts_tag_embedder(torch.tensor([2], dtype=torch.long, device=device))  # [1, D]
        
        # 拼接embeddings
        input_ids_embs = torch.cat([tts_tag_embedder_2, input_ids_embs, tts_tag_embedder_0, global_tokens_embs, tts_tag_embedder_1, semantic_tokens_embs], dim=0)  # [T+1+G+1+S, D]
        input_ids_embs_list.append(input_ids_embs)
        
        # 创建标签
        my_length = input_ids_embs.shape[0]
        my_label = torch.full((my_length,), -100, dtype=torch.long, device=device)
        semantic_length = semantic_tokens.shape[0]
        my_label[-semantic_length-1:-1] = semantic_tokens
        my_label[-1] = eos_token_id
        labels.append(my_label)
        
        # 检查累积序列长度
        last_length = cu_seqlens[-1]
        if last_length + my_length > max_cu_seqlens:
            break
        cu_seqlens.append(last_length + my_length)
    
    # 合并所有embeddings和标签
    input_embs = torch.cat(input_ids_embs_list, dim=0)  # [B, T+1+G+1+S, D]
    labels = torch.cat(labels, dim=0)  # [B, T+1+G+1+S]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.long, device=device)
    
    return {
        "input_embs": input_embs.unsqueeze(0),  # [B, T+1+G+1+S, D]
        "labels": labels.unsqueeze(0),  # [B, T+1+G+1+S]
        "cu_seqlens": cu_seqlens
    }

def process_single_batch(batch, rwkv7speech_model,eos_token_id=8192):
    """
    处理单个batch，生成与collate_fn_for_rwkv7speech相同格式的输出
    
    Args:
        batch: {
                "input_ids": padded_input_ids,
                "global_tokens_ids": padded_global_tokens,
                "semantic_tokens_ids": padded_semantic_tokens,
                "input_ids_len": padded_input_ids_len,
                "global_tokens_len": padded_global_tokens_len,
                "semantic_tokens_len": padded_semantic_tokens_len
            }
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
        text_length = batch['input_ids_len'][i]
        input_ids = batch['input_ids'][i, -text_length:]
        
        # 获取全局标记
        global_length = batch['global_tokens_len'][i]
        global_ids = batch['global_tokens_ids'][i, -global_length:]
        
        # 获取语义标记
        semantic_length = batch['semantic_tokens_len'][i]
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
        semantic_length = batch['semantic_tokens_len'][i]
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


def main():
    """测试 MultipleWebDataset 和 collate_fn_with_bicodec"""
    # 创建数据集实例
    dataset = MultipleWebDataset(
        data_dir="/external_data/yueyudata/voxbox_wids/",
        target_sr=16000,
        target_channels=1,
        shuffle=True
    )
    print(dataset[0])
    rwkv_model_path = "/home/yueyulin/models/rwkv7-0.1B-g1-respark-speech"
    # 创建分词器
    tokenizer = AutoTokenizer.from_pretrained(rwkv_model_path,trust_remote_code=True)
    # 创建 BiCodecTokenizer
    model_dir = '/home/yueyulin/models/Spark-TTS-0.5B/'
    device = 'cuda:3'
    bicodec_tokenizer = BiCodecTokenizer(model_dir, device)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=lambda x: collate_fn_with_bicodec(x, tokenizer, bicodec_tokenizer),
        shuffle=True
    )
    
    rwkv7speech_model = AutoModelForCausalLM.from_pretrained(rwkv_model_path,trust_remote_code=True)
    rwkv7speech_model.to(device)
    print(rwkv7speech_model)
    
    # 测试数据加载
    for batch in dataloader:
        print("\n批次信息:")
        
        # 打印第一个样本的信息
        print("\n第一个样本:")
        print(batch['input_ids'])
        print(batch['global_tokens_ids'])
        print(batch['semantic_tokens_ids'])
        print(batch['texts'])
        for j in range(batch['input_ids'].shape[0]):
            input_ids = batch['input_ids'][j][-batch['input_ids_len'][j]:]
            print(f'decoded text: {tokenizer.decode(input_ids.tolist())}')
        print(batch['global_tokens_ids'].tolist())
        print(batch['semantic_tokens_ids'].tolist())
        training_data = process_single_batch(batch, rwkv7speech_model,eos_token_id=8192)
        print(training_data['input_embs'])
        print(training_data['attention_mask'])
        print(training_data['labels'])
        print(training_data['input_embs'].shape)
        print(training_data['attention_mask'].shape)
        print(training_data['labels'].shape)
        print(training_data['labels'].tolist())
        print(training_data['attention_mask'].tolist())
        print(f"input_ids shape: {batch['input_ids'].shape}")
        print(f"global_tokens_ids shape: {batch['global_tokens_ids'].shape}")
        print(f"semantic_tokens_ids shape: {batch['semantic_tokens_ids'].shape}")

        break

def main_test():
    """测试 MultipleWebDataset 和 collate_fn_with_bicodec"""
    # 创建数据集实例
    dataset = MultipleWebDataset(
        data_dir="/data/training/",
        target_sr=16000,
        target_channels=1,
        shuffle=True
    )
    print(dataset[0])
    rwkv_model_path = "/home/yueyulin/models/rwkv7-0.1B-g1-respark-speech"
    # 创建分词器
    tokenizer = AutoTokenizer.from_pretrained(rwkv_model_path,trust_remote_code=True)
    # 创建 BiCodecTokenizer
    model_dir = '/home/yueyulin/models/Spark-TTS-0.5B/'
    device = 'cuda:3'
    bicodec_tokenizer = BiCodecTokenizer(model_dir, device)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=lambda x: collate_fn_with_tokenizer(x, tokenizer),
        shuffle=True
    )
    
    rwkv7speech_model = AutoModelForCausalLM.from_pretrained(rwkv_model_path,trust_remote_code=True)
    rwkv7speech_model.to(torch.bfloat16).to(device)
    print(rwkv7speech_model)
    from torch.optim import AdamW   
    optimizer = AdamW(rwkv7speech_model.parameters(), lr=1e-4)
    
    rwkv7speech_model.train()
    # 测试数据加载
    for batch in dataloader:
        print("\n批次信息:")
        
        # 打印第一个样本的信息
        print("\n第一个样本:")
        print(batch)
        training_data = process_single_batch_with_audio_tokenizer(batch, rwkv7speech_model,bicodec_tokenizer,eos_token_id=8192)
        print(training_data['input_embs'])
        print(training_data['attention_mask'])
        print(training_data['labels'])
        print(training_data['input_embs'].shape)
        print(training_data['attention_mask'].shape)
        print(training_data['labels'].shape)
        b = training_data['input_embs'].shape[0]
        all_length = 0
        for i in range(b):
            length = training_data['attention_mask'][i].sum().item()
            print(f'length: {length}')
            all_length += length
        print(f'all_length: {all_length}')
        optimizer.zero_grad()
        outputs = rwkv7speech_model.forward(inputs_embeds=training_data['input_embs'],attention_mask=training_data['attention_mask'],labels=training_data['labels'],use_cache=False)
        print(outputs)
        outputs.loss.backward()
        
        # 更新参数
        optimizer.step()

        optimizer.zero_grad()
        print('--------------------------------')
        training_data = process_single_batch_with_audio_tokenizer_culens(batch, rwkv7speech_model,bicodec_tokenizer,eos_token_id=8192)
        b = training_data['cu_seqlens'].shape[0]
        print(training_data['input_embs'])
        print(training_data['labels'])
        print(training_data['cu_seqlens'])
        print(training_data['input_embs'].shape)
        print(training_data['labels'].shape)
        print(training_data['cu_seqlens'].shape)
        prev_len = 0
        for i in range(b):
            length = training_data['cu_seqlens'][i].item()
            print(f'length: {length-prev_len}')
            prev_len = length
        outputs = rwkv7speech_model.forward(inputs_embeds=training_data['input_embs'],labels=training_data['labels'],use_cache=False,cu_seqlens=training_data['cu_seqlens'])
        print(outputs)
        outputs.loss.backward()
        optimizer.step()
        break
if __name__ == "__main__":
    main_test()
