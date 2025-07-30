import lmdb
import os
import json
import click
from voxbox_lmdb_utils import open_lmdb_for_read,get_json_from_lmdb
import hashlib
import glob
from typing import Iterator, Dict, Any

def stream_jsonl_file(jsonl_file: str) -> Iterator[Dict[str, Any]]:
    """
    流式读取JSONL文件，逐行处理避免一次性加载大文件
    
    Args:
        jsonl_file: JSONL文件路径
        
    Yields:
        每行的JSON数据字典
    """
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                data = json.loads(line)
                yield data
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num} in {jsonl_file}: {e}")
                continue

@click.command()
@click.option("--voxbox_lmdb_path", type=str, required=True)
@click.option("--voxbox_jsonl_path", type=str, required=True)
@click.option("--voxbox_output_path", type=str, required=True)
@click.option("--batch_size", type=int, default=1000, help="批处理大小，用于进度显示")
def align_voxbox_properties(voxbox_lmdb_path, voxbox_jsonl_path, voxbox_output_path, batch_size):
    #find the sub directories in voxbox_lmdb_path
    sub_dirs = [d for d in os.listdir(voxbox_lmdb_path) if os.path.isdir(os.path.join(voxbox_lmdb_path, d))]
    environments = {}
    for sub_dir in sub_dirs:
        env = open_lmdb_for_read(os.path.join(voxbox_lmdb_path, sub_dir,'voxbox.lmdb'))
        environments[sub_dir] = env
    #find the sun directories in voxbox_jsonl_path with the same name as the sub directories in voxbox_lmdb_path
    sub_dirs_jsonl = [d for d in os.listdir(voxbox_jsonl_path) if os.path.isdir(os.path.join(voxbox_jsonl_path, d))]
    print(f'found {sub_dirs_jsonl} that have the same name as the sub directories in {voxbox_lmdb_path}')
    for sub_dir_jsonl in sub_dirs_jsonl:
        print(f'aligning {sub_dir_jsonl}')
        if sub_dir_jsonl in environments:
            env = environments[sub_dir_jsonl]
            os.makedirs(os.path.join(voxbox_output_path, sub_dir_jsonl), exist_ok=True)
            jsonl_files = glob.glob(os.path.join(voxbox_jsonl_path, sub_dir_jsonl, '*.jsonl'))
            for jsonl_file in jsonl_files:
                print(f'aligning {jsonl_file}')
                
                # 使用流式加载处理JSONL文件
                processed_count = 0
                found_count = 0
                with open(os.path.join(voxbox_output_path, sub_dir_jsonl, os.path.basename(jsonl_file)), 'w') as f:
                    for data in stream_jsonl_file(jsonl_file):
                        processed_count += 1
                        
                        # 显示进度
                        if processed_count % batch_size == 0:
                            print(f"Processed {processed_count} lines from {jsonl_file}")
                        
                        text = data.get('text', '')
                        if not text or text == "":
                            print(f'Text is None for {data}')
                            continue
                            
                        key = hashlib.md5(text.encode()).hexdigest()
                        lmdb_data = get_json_from_lmdb(env, key)
                        
                        if lmdb_data is not None:
                            json_data = json.loads(lmdb_data)
                            json_data.update(data)
                            f.write(json.dumps(json_data,ensure_ascii=False)+'\n')
                        else:
                            print(f'Text not found in lmdb: {text[:50]}...')
                    
                        print(f"Completed processing {jsonl_file}: {processed_count} lines processed")

if __name__ == "__main__":
    align_voxbox_properties()