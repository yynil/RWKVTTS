import argparse
import os
import json
import numpy as np
import torch
import torchaudio
import soundfile as sf
import multiprocessing as mp
from pathlib import Path
from datasets import load_dataset
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from typing import Dict, Any
import queue
import time
import psutil
import gc  # 添加 gc 模块导入
from datetime import datetime
from utils.RawMutltipleWebDataset import RawMutltipleWebDataset
from tqdm import tqdm
import io

def get_available_gpu(process_id: int):
    """根据进程ID分配GPU，每个GPU最多分配两个进程"""
    try:
        # 获取可用的GPU数量
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return 'cpu'
            
        # 计算应该使用哪个GPU
        gpu_id = (process_id // 2) % device_count
        return f'cuda:{gpu_id}'
    except:
        return 'cpu'

# 特殊标记，用于通知子进程退出
EXIT_SIGNAL = "EXIT"
# 初始化完成信号
INIT_DONE_SIGNAL = "INIT_DONE"

def worker_process(process_id: int, input_queue: mp.Queue, init_done_queue: mp.Queue, output_file: str, model_dir: str):
    """工作进程函数"""
    process = psutil.Process()
    
    # 添加统计信息
    start_time = time.time()
    total_requests = 0
    
    try:
        # 获取GPU资源
        gpu_id = get_available_gpu(process_id)
        print(f"Process {process_id} initializing with device: {gpu_id}")
        
        # 初始化tokenizer
        audio_tokenizer = BiCodecTokenizer(model_dir, device=gpu_id)
        
        # 发送初始化完成信号
        init_done_queue.put(INIT_DONE_SIGNAL)
        print(f"Process {process_id} initialization completed")
        
        # 打开输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            while True:
                try:
                    # 从队列获取数据
                    data = input_queue.get()
                    
                    # 检查是否是退出信号
                    if data == EXIT_SIGNAL:
                        print(f"Process {process_id} received exit signal, shutting down...")
                        break
                    
                    # 解包数据
                    json_data, audio_array, sampling_rate = data
                    
                    # 确保音频数据是float32类型
                    audio_data = np.array(audio_array, dtype=np.float32)
                    target_sample_rate = audio_tokenizer.config['sample_rate']
                    
                    if sampling_rate != target_sample_rate:
                        from librosa import resample
                        audio_data = resample(audio_data, orig_sr=sampling_rate, target_sr=target_sample_rate)
                        audio_data = np.array(audio_data, dtype=np.float32)
                        
                    global_tokens, semantic_tokens = audio_tokenizer.tokenize(audio_data)
                    global_tokens = global_tokens.squeeze(0).squeeze(0).detach().cpu().tolist()
                    semantic_tokens = semantic_tokens.squeeze(0).squeeze(0).detach().cpu().tolist()

                    result = {
                        'language': json_data['language'],
                        'text': json_data['text'],
                        'global_tokens': global_tokens,
                        'semantic_tokens': semantic_tokens
                    }
                    
                    # 写入JSONL文件
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()  # 确保数据及时写入磁盘
                    
                    # 更新统计信息
                    total_requests += 1
                    if total_requests % 1000 == 0:
                        current_time = time.time()
                        total_time = current_time - start_time
                        avg_time = total_time / total_requests
                        print(f"Process {process_id} stats at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:")
                        print(f"  Total requests: {total_requests}")
                        print(f"  Total time: {total_time:.2f}s")
                        print(f"  Average time per request: {avg_time:.2f}s")
                    
                    # 处理完数据后立即清理
                    del audio_array
                    del audio_data
                    del global_tokens
                    del semantic_tokens
                    torch.cuda.empty_cache()  # 清理GPU缓存
                    gc.collect()  # 手动触发垃圾回收
                    
                    # 监控内存使用
                    if process.memory_info().rss > 1024 * 1024 * 1024*100:  # 超过100GB
                        print(f"Process {process_id} memory usage high: {process.memory_info().rss / 1024 / 1024}MB")
                        torch.cuda.empty_cache()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Process {process_id} encountered error: {str(e)}")
                    continue
                    
    except Exception as e:
        print(f"Process {process_id} failed to initialize: {str(e)}")
        # 即使初始化失败也发送信号，避免主进程卡住
        init_done_queue.put(INIT_DONE_SIGNAL)
    finally:
        # 确保在进程结束时清理资源
        del audio_tokenizer
        torch.cuda.empty_cache()
        print(f"Process {process_id} shutting down...")

def data_generator(dataset):
    for item in dataset:
        yield (item['json'], item['mp3']['array'], item['mp3']['sampling_rate'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/tmp/tmp_data/')
    parser.add_argument('--output_dir', type=str, default='/home/yueyulin/data/Emilia/ZH/tar_tokens/')
    parser.add_argument('--model_dir', type=str, default='/home/yueyulin/models/Spark-TTS-0.5B/')
    parser.add_argument('--num_proc', type=int, default=4, help='Number of processes to use')
    parser.add_argument('--from_index', type=int, default=0, help='Start index of files to process (inclusive)')
    parser.add_argument('--to_index', type=int, default=None, help='End index of files to process (exclusive)')
    args = parser.parse_args()

    print(args)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建进程间通信队列
    queues = [mp.Queue() for _ in range(args.num_proc)]
    init_done_queues = [mp.Queue() for _ in range(args.num_proc)]
    
    # 创建并启动工作进程
    processes = []
    dir_name = os.path.basename(os.path.normpath(args.input_dir))
    for i in range(args.num_proc):
        output_file = os.path.join(args.output_dir, f'{dir_name}_{i}.jsonl')
        p = mp.Process(
            target=worker_process,
            args=(i, queues[i], init_done_queues[i], output_file, args.model_dir)
        )
        p.start()
        processes.append(p)
        
        # 等待当前进程初始化完成
        init_signal = init_done_queues[i].get()
        if init_signal != INIT_DONE_SIGNAL:
            print(f"Warning: Process {i} initialization signal unexpected: {init_signal}")
    
    print("All processes initialized successfully")
    import glob
    all_tars = glob.glob(os.path.join(args.input_dir, "*.tar"))
    #sort by name
    all_tars.sort()
    if args.to_index is None:
        args.to_index = len(all_tars)
        print(f"Using all files from index {args.from_index} to {args.to_index}")
    # 使用from_index和to_index来切片文件列表
    all_tars = all_tars[args.from_index:args.to_index]
    print(f"Processing {len(all_tars)} files from index {args.from_index} to {args.to_index}")
    try:
        # 加载数据集
        dataset = RawMutltipleWebDataset(data_files=all_tars)
        
        current_queue_idx = 0
        for i, item in enumerate(tqdm(dataset, desc=f"Processing Tars from {dir_name}")):
            json_data = json.loads(item['json'])
            audio_buffer = io.BytesIO(item['wav'])
            audio_tensor2, sample_rate2 = torchaudio.load(audio_buffer)
            if audio_tensor2.shape[0] > 1:
                # 将多声道音频转换为单声道
                audio_tensor2 = audio_tensor2.mean(dim=0)
            audio_array = audio_tensor2.tolist()
            data = (json_data, audio_array, sample_rate2)
            
            # 轮询查找空闲队列
            max_attempts = args.num_proc
            attempts = 0
            while attempts < max_attempts:
                if queues[current_queue_idx].qsize() < 100:  # 队列未满
                    queues[current_queue_idx].put(data)
                    current_queue_idx = (current_queue_idx + 1) % args.num_proc
                    break
                else:
                    current_queue_idx = (current_queue_idx + 1) % args.num_proc
                    attempts += 1
                    if attempts == max_attempts:
                        time.sleep(1)
                        attempts = 0
            
            # 定期清理内存
            if i % 100 == 0:  # 每处理100个文件清理一次
                torch.cuda.empty_cache()
                gc.collect()  # 手动触发垃圾回收
        
        # 发送退出信号给所有进程
        for queue in queues:
            queue.put(EXIT_SIGNAL)
        
        # 等待所有进程完成
        for p in processes:
            p.join()
            
    except KeyboardInterrupt:
        print("Received keyboard interrupt, shutting down...")
        # 发送退出信号给所有进程
        for queue in queues:
            queue.put(EXIT_SIGNAL)
        # 等待所有进程完成
        for p in processes:
            p.join()
    
    print("All processes completed.")

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()