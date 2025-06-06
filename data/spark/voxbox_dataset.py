import os
import json
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import logging
from sparktts.models.audio_tokenizer import BiCodecTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class VoxBoxDataset(Dataset):
    """VoxBox数据集加载器
    
    该数据集类用于加载VoxBox格式的语音数据集，支持自动提取音频token。
    
    Args:
        root_dir (str): 数据集根目录，包含audios和metadata子目录
        model_dir (str): BiCodec模型目录
        target_sr (int): 目标采样率，默认16000
        target_channels (int): 目标声道数，默认1（单声道）
        split (Optional[str]): 数据集划分，可选'train'/'test'/'val'，默认None表示加载所有数据
    """
    
    def __init__(
        self,
        root_dir: str,
        model_dir: str,
        target_sr: int = 16000,
        target_channels: int = 1,
        split: Optional[str] = None
    ):
        self.root_dir = root_dir
        self.target_sr = target_sr
        self.target_channels = target_channels
        self.split = split
        
        self.audio_dir = os.path.join(root_dir, "audios")
        self.metadata_dir = os.path.join(root_dir, "metadata")
        
        # 初始化BiCodec模型
        self.audio_tokenizer = BiCodecTokenizer(model_dir, device='cpu')
        
        # 获取所有可用的数据集
        self.datasets = self._get_available_datasets()
        
        # 加载所有元数据
        self.metadata = self._load_metadata()
        
        logger.info(f"成功加载数据集，共 {len(self.metadata)} 条数据")
        
    def _get_available_datasets(self) -> List[str]:
        """获取所有可用的数据集名称"""
        datasets = []
        for jsonl_file in os.listdir(self.metadata_dir):
            if jsonl_file.endswith('.jsonl'):
                dataset_name = jsonl_file[:-6]  # 移除.jsonl后缀
                if os.path.exists(os.path.join(self.audio_dir, dataset_name)):
                    datasets.append(dataset_name)
        return datasets
    
    def _load_metadata(self) -> List[Dict]:
        """加载所有元数据文件"""
        all_metadata = []
        for dataset in self.datasets:
            jsonl_path = os.path.join(self.metadata_dir, f"{dataset}.jsonl")
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if self.split is None or data['split'] == self.split:
                        all_metadata.append(data)
        return all_metadata
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取单个数据样本
        
        Returns:
            Dict: 包含以下字段的字典：
                - global_tokens: 全局音频token
                - semantic_tokens: 语义音频token
                - text: 文本内容
        """
        item = self.metadata[idx]
        audio_path = os.path.join(self.audio_dir, item['wav_path'])
        
        # 加载音频
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 标准化音频格式
        if sample_rate != self.target_sr:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sr
            )(waveform)
        
        if waveform.shape[0] != self.target_channels:
            if waveform.shape[0] > self.target_channels:
                waveform = waveform[:self.target_channels]
            else:
                waveform = torch.cat([waveform] * (self.target_channels // waveform.shape[0]))
        
        # 转换为numpy数组
        audio_data = waveform.squeeze().numpy().astype(np.float32)
        
        # 提取token
        global_tokens, semantic_tokens = self.audio_tokenizer.tokenize(audio_data)
        global_tokens = global_tokens.squeeze(0).squeeze(0).detach().cpu().tolist()
        semantic_tokens = semantic_tokens.squeeze(0).squeeze(0).detach().cpu().tolist()
        
        return {
            'global_tokens': global_tokens,
            'semantic_tokens': semantic_tokens,
            'text': item['text']
        }

if __name__ == "__main__":
    root_dir = os.path.expanduser("~/data/voxbox_replica")
    model_dir = "/home/yueyulin/models/Spark-TTS-0.5B/"  # 请替换为实际的模型目录
    dataset = VoxBoxDataset(
            root_dir=root_dir,
            model_dir=model_dir,
            target_sr=16000,
            target_channels=1
        )
    print(dataset[0])
    from data.utils.spark_dataset import collate_fn_simple
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from functools import partial
    from torch.optim import AdamW
    from data.utils.spark_dataset import process_single_batch
    device = "cuda:0"
    rwkv_model_dir = '/home/yueyulin/models/rwkv7-0.1B-g1-respark-speech/'
    tokenizer = AutoTokenizer.from_pretrained(rwkv_model_dir,trust_remote_code=True)
    rwkv7speech_model = AutoModelForCausalLM.from_pretrained(rwkv_model_dir,trust_remote_code=True).bfloat16().to(device)
    rwkv7speech_model.train()
    dataloader = DataLoader(dataset,batch_size=2,shuffle=True,collate_fn=partial(collate_fn_simple,tokenizer=tokenizer),num_workers=4)

    optimizer = AdamW(rwkv7speech_model.parameters(), lr=1e-4)
    rwkv7speech_model.gradient_checkpointing_enable()
    for batch in dataloader:
        print(batch)
        print("--------------------------------")
        processed_batch = process_single_batch(batch, rwkv7speech_model)
        # print(processed_batch['labels'].tolist())
        print(f'input_embs shape:{processed_batch["input_embs"].shape}, attention_mask shape:{processed_batch["attention_mask"].shape}, labels shape:{processed_batch["labels"].shape}')
        outputs = rwkv7speech_model.forward(inputs_embeds=processed_batch['input_embs'],attention_mask=processed_batch['attention_mask'],labels=processed_batch['labels'],use_cache=False)
        print(outputs)
        outputs.loss.backward()
        
        # 更新参数
        optimizer.step()
        optimizer.zero_grad()
        break