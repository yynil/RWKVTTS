import os
import torch
import torchaudio
import random
from multiple_webdataset import MultipleWebDataset
from pathlib import Path

def save_audio(audio_data, sample_rate, filename):
    """保存音频文件"""
    # 确保音频数据是 tensor 格式
    if not isinstance(audio_data, torch.Tensor):
        audio_data = torch.tensor(audio_data)
    # 确保音频数据有正确的维度
    if audio_data.dim() == 1:
        audio_data = audio_data.unsqueeze(0)
    torchaudio.save(filename, audio_data, sample_rate)

def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    random.seed(42)
    
    # 创建数据集实例
    dataset = MultipleWebDataset(
        data_dir="/external_data/yueyudata/voxbox_wids/",
        target_sr=16000,
        target_channels=1,
        shuffle=True
    )
    
    # 创建输出目录
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # 直接获取10条数据
    print("开始加载数据...")
    for i in range(10):
        try:
            # 获取数据
            sample = dataset[i]
            
            # 打印样本信息
            print(f"\n样本 {i+1}:")
            print(f"音频数据: {sample['audio']['array']}")
            print(f"采样率: {sample['audio']['sampling_rate']}")
            print(f"音频路径: {sample['audio']['path']}")
            
            # 保存音频文件
            output_file = output_dir / f"sample_{i+1}.wav"
            save_audio(
                sample['audio']['array'],
                sample['audio']['sampling_rate'],
                str(output_file)
            )
            print(f"已保存到: {output_file}")
            
            # 打印元数据信息
            if 'json' in sample:
                print("元数据信息:")
                for key, value in sample['json'].items():
                    print(f"  {key}: {value}")
                
        except Exception as e:
            print(f"处理样本 {i} 时出错: {str(e)}")
            continue
            
    print(f"\n当前已加载的样本数量: {len(dataset)}")

if __name__ == "__main__":
    main() 