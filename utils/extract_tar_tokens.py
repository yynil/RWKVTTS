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
                        'speaker': json_data['speaker'],
                        'global_tokens': global_tokens,
                        'semantic_tokens': semantic_tokens
                    }
                    
                    # 写入JSONL文件
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()  # 确保数据及时写入磁盘
                    
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
        print(f"Process {process_id} shutting down...")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/tmp/tmp_data/')
    parser.add_argument('--output_dir', type=str, default='/home/yueyulin/data/Emilia/ZH/tar_tokens/')
    parser.add_argument('--model_dir', type=str, default='/home/yueyulin/models/Spark-TTS-0.5B/')
    parser.add_argument('--num_proc', type=int, default=4, help='Number of processes to use')
    args = parser.parse_args()

    print(args)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建进程间通信队列
    queues = [mp.Queue() for _ in range(args.num_proc)]
    init_done_queues = [mp.Queue() for _ in range(args.num_proc)]
    
    # 创建并启动工作进程
    processes = []
    for i in range(args.num_proc):
        output_file = os.path.join(args.output_dir, f'output_{i}.jsonl')
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
    
    try:
        # 加载数据集
        dataset = load_dataset('webdataset', data_files=os.path.join(args.input_dir, '*.tar'), split='train')
        
        # 分发数据到工作进程
        for i, item in enumerate(dataset):
            queue_idx = i % args.num_proc
            # 只传递必要的数据
            data = (item['json'], item['mp3']['array'], item['mp3']['sampling_rate'])
            queues[queue_idx].put(data)
        
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
    main()
    input('Press Enter to continue...')