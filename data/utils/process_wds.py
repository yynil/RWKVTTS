import os
import json
import torch
import webdataset as wds
import argparse
from tqdm import tqdm
import torchaudio
import io

def decode_audio(data):
    """解码音频数据"""
    try:
        # 将字节数据转换为文件对象
        audio_file = io.BytesIO(data)
        # 使用torchaudio加载音频
        waveform, sample_rate = torchaudio.load(audio_file)
        return waveform
    except Exception as e:
        print(f"Error decoding audio: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Process WebDataset format data')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing WebDataset tar files')
    args = parser.parse_args()

    # 获取所有tar文件
    tar_files = sorted([
        os.path.join(args.input_dir, f) 
        for f in os.listdir(args.input_dir) 
        if f.endswith('.tar')
    ])
    
    print(f"Found {len(tar_files)} tar files")
    
    # 创建数据集
    dataset = wds.WebDataset(tar_files)
    
    # 定义数据处理管道
    dataset = dataset.map_dict(
        flac=decode_audio,  # 使用自定义函数解码音频
        json=lambda x: x if isinstance(x, dict) else json.loads(x.decode('utf-8'))  # 处理JSON数据
    )
    
    # 测试数据集
    print("Testing dataset...")
    try:
        # 获取第一个样本
        sample = next(iter(dataset))
        print("\nSample structure:")
        for k, v in sample.items():
            print(f"{k}: {type(v)}")
            if isinstance(v, torch.Tensor):
                print(f"  Shape: {v.shape}")
            elif isinstance(v, dict):
                print(f"  Keys: {list(v.keys())}")
                
    except Exception as e:
        print(f"Error testing dataset: {str(e)}")

if __name__ == '__main__':
    main() 