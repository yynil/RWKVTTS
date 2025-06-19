#!/usr/bin/env python3
import os
import tarfile
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.data import DataLoader
from functools import partial

def check_tar_file(tar_path):
    """检查单个tar文件的完整性"""
    try:
        with tarfile.open(tar_path, 'r') as tar:
            # 尝试读取所有成员的信息
            tar.getmembers()
        return True, tar_path
    except Exception as e:
        return False, f"{tar_path}: {str(e)}"

def main():
    # 设置要检查的目录
    base_dirs = [
        "/data/training/emilia_en",
        "/data/training/emilia_zh"
    ]
    
    all_tar_files = []
    for base_dir in base_dirs:
        tar_files = list(Path(base_dir).glob("*.tar"))
        all_tar_files.extend(tar_files)
    
    print(f"找到 {len(all_tar_files)} 个tar文件需要检查")
    
    # 使用线程池并行检查文件
    corrupted_files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # 使用tqdm显示进度条
        futures = [executor.submit(check_tar_file, str(tar_path)) for tar_path in all_tar_files]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="检查文件"):
            is_valid, result = future.result()
            if not is_valid:
                corrupted_files.append(result)
    
    # 输出结果
    if corrupted_files:
        print("\n发现损坏的文件：")
        for file in corrupted_files:
            print(file)
    else:
        print("\n所有文件都是完整的！")

if __name__ == "__main__":
    # main() 
    multiple_webdataset_dir = "/data/training/"
    import glob
    tar_files = glob.glob(os.path.join(multiple_webdataset_dir, "**/*.tar"))
    from datasets import load_dataset
    model_name = "/home/yueyulin/models/rwkv7-0.1B-g1-respark-speech/"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    from data.spark.multiple_webdataset import MultipleWebDataset, collate_fn_with_tokenizer
    from torch.utils.data import DataLoader
    data_collator = partial(collate_fn_with_tokenizer,  
                            tokenizer=tokenizer)
    for tar_file in tar_files:
        ds = MultipleWebDataset(
            data_dir=tar_file,
            target_sr=16000,
            target_channels=1,
            shuffle=False,
            verify_tar=False
        )
        
        progress_bar = tqdm(len(ds), desc=f"Processing {tar_file}")
        for batch in progress_bar:
            audios = batch["audios"]
            print(audios)
            break