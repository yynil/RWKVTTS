import os
import json
import gzip
import tarfile
import io
import shutil
import argparse
from tqdm import tqdm
import time
import mmap
import psutil
import gc
from multiprocessing import Pool, cpu_count
from functools import partial

def get_memory_usage():
    """获取当前进程的内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # 转换为MB

def get_system_memory():
    """获取系统内存信息"""
    mem = psutil.virtual_memory()
    return {
        'total': mem.total / 1024 / 1024 / 1024,  # GB
        'available': mem.available / 1024 / 1024 / 1024,  # GB
        'percent': mem.percent
    }

def process_tar_file(input_file, input_dir, output_dir, metadata_map):
    """处理单个tar.gz文件的函数"""
    input_path = os.path.join(input_dir, input_file)
    output_file = input_file.replace('.tar.gz', '.tar')
    output_path = os.path.join(output_dir, output_file)
    
    try:
        # 使用mmap加载整个文件到内存
        with open(input_path, 'rb') as f_in:
            # 获取文件大小
            file_size = os.path.getsize(input_path)
            print(f"Processing {input_file} ({file_size/1024/1024:.1f}MB)...")
            
            # 使用mmap加载文件
            with mmap.mmap(f_in.fileno(), 0, access=mmap.ACCESS_READ) as mm_in:
                # 直接使用gzip.open处理文件
                with gzip.open(input_path, 'rb') as gz_file, \
                     tarfile.open(fileobj=gz_file, mode='r:') as in_tar, \
                     tarfile.open(output_path, 'w') as out_tar:
                    
                    sample_count = 0
                    file_count = 0
                    last_print_time = time.time()
                    print_interval = 1000  # 每处理1000个文件打印一次
                    
                    # 使用迭代器顺序处理文件
                    for member in in_tar:
                        file_count += 1
                        if not member.isfile():
                            continue
                        
                        # 从文件名提取索引
                        file_key = os.path.splitext(os.path.basename(member.name))[0]
                        original_format = os.path.splitext(member.name)[1][1:].lower()
                        
                        # 获取对应的元数据
                        if file_key not in metadata_map:
                            print(f"  WARNING: Metadata missing for {file_key}")
                            continue
                        
                        meta = metadata_map[file_key]
                        
                        # 读取音频内容
                        audio_data = in_tar.extractfile(member).read()
                        
                        # 准备样本编号 (6位数字)
                        sample_id = f"{sample_count:06d}"
                        
                        # 添加到WebDataset
                        # 1. 添加音频文件（保持原始格式）
                        audio_info = tarfile.TarInfo(f"{sample_id}.{original_format}")
                        audio_info.size = len(audio_data)
                        out_tar.addfile(audio_info, io.BytesIO(audio_data))
                        
                        # 2. 添加JSON元数据
                        json_data = json.dumps(meta).encode('utf-8')
                        json_info = tarfile.TarInfo(f"{sample_id}.json")
                        json_info.size = len(json_data)
                        out_tar.addfile(json_info, io.BytesIO(json_data))
                        
                        sample_count += 1
                        
                        # 定期打印进度
                        if sample_count % print_interval == 0:
                            current_time = time.time()
                            elapsed_time = current_time - last_print_time
                            files_per_second = print_interval / elapsed_time
                            print(f"  [{input_file}] Processed {sample_count} files, {files_per_second:.1f} files/sec")
                            last_print_time = current_time
                    
                    print(f"  Completed {input_file}: Created {sample_count} samples")
                    
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Convert dataset to WebDataset format')
    parser.add_argument('--metadata', type=str, required=True, help='Path to metadata JSONL file')
    parser.add_argument('--input_dir', type=str, required=True, help='Base directory containing input tar.gz files')
    parser.add_argument('--output_dir', type=str, required=True, help='Base output directory for WebDataset tar files')
    parser.add_argument('--chunk_size', type=int, default=1024*1024*64, help='Chunk size for file operations (default: 64MB)')
    parser.add_argument('--max_memory_gb', type=int, default=100, help='Maximum memory usage in GB (default: 100)')
    parser.add_argument('--num_processes', type=int, default=4, help='Number of processes to use (default: CPU count)')
    args = parser.parse_args()

    # 从metadata路径中提取数据集名称
    dataset_name = os.path.splitext(os.path.basename(args.metadata))[0]
    input_dir = os.path.join(args.input_dir, dataset_name)
    output_dir = os.path.join(args.output_dir, dataset_name)
    
    print(f"Dataset name: {dataset_name}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # 检查系统内存
    system_mem = get_system_memory()
    print(f"System memory: {system_mem['total']:.1f}GB total, {system_mem['available']:.1f}GB available")
    print(f"Initial memory usage: {get_memory_usage():.2f} MB")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 步骤1: 使用mmap加载元数据到内存
    metadata_map = {}
    print(f"Loading metadata from {args.metadata}...")
    
    with open(args.metadata, 'r') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for line in tqdm(iter(mm.readline, b''), desc="Loading metadata"):
                try:
                    data = json.loads(line.decode('utf-8'))
                    index = data['index']
                    metadata_map[index] = data
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON line: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line: {e}")
                    continue
    
    print(f"Loaded {len(metadata_map)} metadata entries")
    print(f"Memory usage after loading metadata: {get_memory_usage():.2f} MB")

    # 步骤2: 获取所有输入tar.gz文件
    input_files = sorted(
        [f for f in os.listdir(input_dir) if f.endswith('.tar.gz')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    print(f"Found {len(input_files)} tar files to process")

    # 步骤3: 使用多进程处理文件
    start_time = time.time()
    
    # 确定进程数
    num_processes = args.num_processes if args.num_processes is not None else cpu_count()
    print(f"Using {num_processes} processes")
    
    # 创建进程池
    with Pool(num_processes) as pool:
        # 创建偏函数，固定metadata_map参数
        process_func = partial(process_tar_file, input_dir=input_dir, output_dir=output_dir, metadata_map=metadata_map)
        
        # 使用tqdm显示进度
        results = list(tqdm(
            pool.imap(process_func, input_files),
            total=len(input_files),
            desc="Processing files"
        ))

    # 统计处理结果
    successful = sum(1 for r in results if r)
    failed = len(results) - successful
    
    total_time = time.time() - start_time
    print(f"\nConversion completed in {total_time/60:.1f} minutes!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed to process: {failed} files")
    print(f"Final memory usage: {get_memory_usage()/1024:.1f}GB")

if __name__ == '__main__':
    main()