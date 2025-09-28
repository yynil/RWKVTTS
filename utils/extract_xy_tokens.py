import argparse
import os
import json
import numpy as np
import torch
import multiprocessing as mp
from pathlib import Path
import queue
import time
import psutil
import gc
from datetime import datetime
from tqdm import tqdm

# Project-specific imports
from XY_Tokenizer.xy_tokenizer.model import XY_Tokenizer
from data.spark.multiple_webdataset import MultipleWebDataset

def get_available_gpu(process_id: int):
    """
    Assigns a GPU to a process based on its ID.
    Each GPU is assigned to a maximum of two processes.
    """
    try:
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return 'cpu'
        gpu_id = (process_id // 2) % device_count
        return f'cuda:{gpu_id}'
    except Exception:
        return 'cpu'

EXIT_SIGNAL = "EXIT"
INIT_DONE_SIGNAL = "INIT_DONE"

def worker_process(process_id: int, input_queue: mp.Queue, init_done_queue: mp.Queue, output_file: str, config_path: str, ckpt_path: str):
    """
    Worker process for tokenizing audio files.
    """
    process = psutil.Process()
    start_time = time.time()
    total_requests = 0

    try:
        gpu_id = get_available_gpu(process_id)
        print(f"Process {process_id} initializing with device: {gpu_id}")
        device = torch.device(gpu_id)

        # Initialize XY_Tokenizer
        xy_tokenizer = XY_Tokenizer.load_from_checkpoint(config_path, ckpt_path)
        xy_tokenizer.eval().to(device)
        target_sample_rate = 16000  # As in train_xy_llm.py

        init_done_queue.put(INIT_DONE_SIGNAL)
        print(f"Process {process_id} initialization completed")

        with open(output_file, 'w', encoding='utf-8') as f:
            while True:
                try:
                    data = input_queue.get()

                    if data == EXIT_SIGNAL:
                        print(f"Process {process_id} received exit signal, shutting down...")
                        break

                    json_data, audio_array, sampling_rate = data
                    
                    audio_data = np.array(audio_array, dtype=np.float32)

                    if sampling_rate != target_sample_rate:
                        from librosa import resample
                        audio_data = resample(y=audio_data, orig_sr=sampling_rate, target_sr=target_sample_rate)
                    
                    with torch.no_grad():
                        audio_tensor = torch.from_numpy(audio_data).to(device)
                        
                        # The tokenizer expects a list of tensors
                        encoded_audio = xy_tokenizer.encode([audio_tensor], device=device)
                        # The output is a 2D array [8, T]
                        speech_tokens = encoded_audio['codes_list'][0]
                        speech_tokens = speech_tokens.detach().cpu().tolist()

                    result = {
                        'audio_tokens': speech_tokens
                    }
                    result.update(json_data)

                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()

                    total_requests += 1
                    if total_requests % 1000 == 0:
                        current_time = time.time()
                        total_time = current_time - start_time
                        avg_time = total_time / total_requests if total_requests > 0 else 0
                        print(f"Process {process_id} stats at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:")
                        print(f"  Total requests: {total_requests}")
                        print(f"  Total time: {total_time:.2f}s")
                        print(f"  Average time per request: {avg_time:.2f}s")

                    del json_data, audio_array, audio_data, speech_tokens, audio_tensor, result
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                    if process.memory_info().rss > 100 * 1024 * 1024 * 1024:  # 100GB
                        print(f"Process {process_id} memory usage high: {process.memory_info().rss / (1024**2):.2f}MB")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                except queue.Empty:
                    continue
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Process {process_id} encountered error: {str(e)}")
                    continue

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Process {process_id} failed to initialize: {str(e)}")
        init_done_queue.put(INIT_DONE_SIGNAL)
    finally:
        if 'xy_tokenizer' in locals():
            del xy_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Process {process_id} shutting down...")

def main():
    parser = argparse.ArgumentParser(description="Extract audio tokens using XY_Tokenizer from webdataset tars.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing webdataset .tar files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output .jsonl files.')
    parser.add_argument('--xy_tokenizer_config_path', type=str, required=True, help='Path to the XY_Tokenizer config YAML file.')
    parser.add_argument('--xy_tokenizer_ckpt_path', type=str, required=True, help='Path to the XY_Tokenizer checkpoint file.')
    parser.add_argument('--num_proc', type=int, default=4, help='Number of processes to use.')
    parser.add_argument('--from_index', type=int, default=0, help='Start index of files to process (inclusive).')
    parser.add_argument('--to_index', type=int, default=None, help='End index of files to process (exclusive).')
    args = parser.parse_args()

    print(args)

    os.makedirs(args.output_dir, exist_ok=True)

    queues = [mp.Queue(maxsize=200) for _ in range(args.num_proc)]
    init_done_queues = [mp.Queue() for _ in range(args.num_proc)]

    processes = []
    dir_name = os.path.basename(os.path.normpath(args.input_dir))
    for i in range(args.num_proc):
        output_file = os.path.join(args.output_dir, f'{dir_name}_{i}.jsonl')
        p = mp.Process(
            target=worker_process,
            args=(i, queues[i], init_done_queues[i], output_file, args.xy_tokenizer_config_path, args.xy_tokenizer_ckpt_path)
        )
        p.start()
        processes.append(p)

    for i in range(args.num_proc):
        init_signal = init_done_queues[i].get()
        if init_signal != INIT_DONE_SIGNAL:
            print(f"Warning: Process {i} initialization signal unexpected: {init_signal}")

    print("All processes initialized successfully")
    
    import glob
    all_tars = sorted(glob.glob(os.path.join(args.input_dir, '*.tar')))
    
    if args.to_index is None:
        args.to_index = len(all_tars)
    
    all_tars = all_tars[args.from_index:args.to_index]
    
    print(f"Processing {len(all_tars)} files from index {args.from_index} to {args.to_index}")

    try:
        dataset = MultipleWebDataset(data_files=all_tars, target_sr=16000, target_channels=1, shuffle=False, verify_tar=False)
        
        current_queue_idx = 0
        for i, item in enumerate(tqdm(dataset, desc=f"Processing Tars from {dir_name}")):
            if 'audio' not in item or 'array' not in item['audio'] or 'sampling_rate' not in item['audio']:
                print(f"Skipping item due to missing audio data: {item.get('__key__')}")
                continue
            
            data = (item['json'], item['audio']['array'], item['audio']['sampling_rate'])
            
            # Put data into queue, wait if full
            queues[current_queue_idx].put(data)
            current_queue_idx = (current_queue_idx + 1) % args.num_proc
            
            if i % 500 == 0:
                gc.collect()

        for queue in queues:
            queue.put(EXIT_SIGNAL)

        for p in processes:
            p.join()

    except KeyboardInterrupt:
        print("Received keyboard interrupt, shutting down...")
        for p in processes:
            p.terminate()
            p.join()

    print("All processes completed.")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
