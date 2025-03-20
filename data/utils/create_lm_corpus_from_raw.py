import os
import numpy as np
import pandas as pd
import json
import io
import torch
import soundfile as sf
import pyarrow.parquet as pq
import whisper
from librosa import resample
import multiprocessing
from tqdm import tqdm
import onnxruntime
from onnxruntime import InferenceSession

def process_file(file_info):
    """处理单个parquet文件的函数，每个进程调用一次"""
    parquet_file, output_path, speech_tokenizer_model, device = file_info
    
    # 为每个进程创建独立的speech_tokenizer_session
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    cuda_idx = int(device.split(':')[-1] if device is not None and 'cuda' in device else '0')
    speech_tokenizer_session = InferenceSession(speech_tokenizer_model, sess_options=option,
                                               providers=[("CUDAExecutionProvider", {"device_id": cuda_idx}) 
                                                         if torch.cuda.is_available() else "CPUExecutionProvider"])
    
    try:
        # 创建目标文件名
        base_filename = os.path.splitext(os.path.basename(parquet_file))[0]
        output_file = os.path.join(output_path, f"{base_filename}_tokens.jsonl")
        
        # 使用PyArrow读取parquet文件的元数据，获取总行数
        parquet_metadata = pq.read_metadata(parquet_file)
        total_rows = parquet_metadata.num_rows
        batch_size = 1000
        
        # 检查是否有已经处理过的文件，计算已处理的行数
        processed_rows = 0
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f_check:
                for _ in f_check:
                    processed_rows += 1
            print(f"Found existing file {output_file} with {processed_rows} processed rows")
        
        # 如果已经处理完所有行，跳过此文件
        if processed_rows >= total_rows:
            return f"Skipped {parquet_file}: all {total_rows} rows already processed"
        
        # 逐批处理数据，以追加方式打开输出文件
        with open(output_file, 'a' if processed_rows > 0 else 'w', encoding='utf-8') as f_out:
            pf = pq.ParquetFile(parquet_file)
            progress = tqdm(total=total_rows, initial=processed_rows, 
                           desc=f"Processing {os.path.basename(parquet_file)}", 
                           position=multiprocessing.current_process()._identity[0] % 10)
            
            skip_rows = processed_rows
            current_row = 0
            
            for batch in pf.iter_batches(batch_size=batch_size):
                df_batch = batch.to_pandas()
                
                # 处理当前批次中的每一行
                for _, row in df_batch.iterrows():
                    current_row += 1
                    
                    # 跳过已处理的行
                    if current_row <= skip_rows:
                        continue
                    
                    audio_obj = row['audio']
                    audio_data = audio_obj['bytes']
                    transcription = row['transcription']
                    language = row['language']
                    speaker = row['speaker']
                    
                    with io.BytesIO(audio_data) as buffer:
                        prompt_data, sample_rate = sf.read(buffer)
                        # 确保是单声道，并转换为float32
                        if len(prompt_data.shape) > 1:
                            prompt_data = prompt_data[:, 0]
                        prompt_data = prompt_data.astype(np.float32)
                                
                        # 重采样到16kHz (如果需要)
                        if sample_rate != 16000:
                            prompt_data = resample(prompt_data, orig_sr=sample_rate, target_sr=16000)
                                
                        prompt_speech_16k = torch.tensor(prompt_data).unsqueeze(0)

                    feat = whisper.log_mel_spectrogram(prompt_speech_16k, n_mels=128)
                    speech_token = speech_tokenizer_session.run(None,
                                                       {speech_tokenizer_session.get_inputs()[0].name:
                                                        feat.detach().cpu().numpy(),
                                                        speech_tokenizer_session.get_inputs()[1].name:
                                                        np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
                    
                    # 写入结果
                    f_out.write(json.dumps({'tts_speech_tokens':speech_token,
                                           'text':transcription,
                                           'language':language,
                                           'speaker':speaker,
                                           "prompt_text":"",
                                           "llm_prompt_speech_token":[]},
                                          ensure_ascii=False)+'\n')
                    progress.update(1)
                
                # 释放内存
                del df_batch
                import gc
                gc.collect()
        
        return f"Successfully processed {parquet_file}: {total_rows-processed_rows} new rows processed"
    except Exception as e:
        return f"Error processing {parquet_file}: {str(e)}"

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/external_data/yueyudata/starrail-voice')
    parser.add_argument('--output_path',type=str,default='/external_data/yueyudata/starrail-voice-voice_tokens')
    parser.add_argument('--speech_tokenizer_model',type=str,default='/external_data/models/CosyVoice2-0.5B_RWKV_1.5B/speech_tokenizer_v2.onnx')
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument('--num_processes',type=int,default=4)
    args = parser.parse_args()
    
    data_path = args.data_path
    output_path = args.output_path
    device = args.device
    speech_tokenizer_model = args.speech_tokenizer_model
    num_processes = args.num_processes
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 找到所有parquet文件
    parquet_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    print(f'Found {len(parquet_files)} parquet files in {data_path}')
    
    # 准备多进程参数
    file_info_list = [(file, output_path, speech_tokenizer_model, device) for file in parquet_files]
    
    # 使用进程池处理文件
    print(f"Starting processing with {num_processes} processes")
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_file, file_info_list)
    
    # 输出处理结果
    for result in results:
        print(result)
    
    print("All files processed successfully!")