from webdataset import WebDataset
from torch.utils.data import Dataset

class RawMutltipleWebDataset(Dataset):
    def __init__(self, data_files):
        self.data_files = sorted(data_files)
        self.current_file_index = 0
        self.wds = WebDataset(self.data_files)

    def __iter__(self):
        for item in self.wds:
            yield item

    def __len__(self):
        return len(self.wds)

    def __getitem__(self, index):
        return self.wds[index]
    
if __name__ == "__main__":
    dir = "/home/yueyulin/data/bili-webdataset"
    import glob
    import os
    import json
    import torchaudio
    import torch
    data_files = glob.glob(os.path.join(dir, "*.tar"))
    print(data_files)
    dataset = RawMutltipleWebDataset(data_files)
    i = 0
    for item in dataset:
        i += 1
        if i < 100:
            continue
        json_data = json.loads(item['json'])
        wav_data = item['wav']
        print(json_data)
        print(len(wav_data))
        # wav_data 是原始的 wav 文件字节数据，需要先保存为临时文件然后用 torchaudio 读取
        import tempfile
        import io
        
        # 方法2: 使用 io.BytesIO (如果上面的方法有问题)
        try:
            audio_buffer = io.BytesIO(wav_data)
            audio_tensor2, sample_rate2 = torchaudio.load(audio_buffer)
            print(f"方法2 - 读取的音频形状: {audio_tensor2.shape}")
            print(f"方法2 - 采样率: {sample_rate2}")
            
            output_path2 = "output_audio_method2.wav"
            torchaudio.save(output_path2, audio_tensor2, sample_rate2)
            print(f"方法2 - 音频已保存到: {output_path2}")
            
        except Exception as e:
            print(f"方法2失败: {e}")
        

        break