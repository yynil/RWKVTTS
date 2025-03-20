import os
from re import A
import whisper
from librosa import resample
import multiprocessing
from tqdm import tqdm
import onnxruntime
from onnxruntime import InferenceSession
import torch
import pyarrow.parquet as pq
import numpy as np
import json
import io
import soundfile as sf
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import mmap
import os
import pyarrow.parquet as pq
import io
import soundfile as sf
import torchaudio.compliance.kaldi as kaldi
import torch
import numpy as np
import onnxruntime

def process_file(file_info):
    """处理单个parquet文件的函数，每个进程调用一次"""
    parquet_file, output_path, speaker_extractor, device = file_info
    
    # 为每个进程创建独立的speech_tokenizer_session
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    ort_session = onnxruntime.InferenceSession(speaker_extractor, sess_options=option,
                                               providers=["CPUExecutionProvider"])
    results = {}
    try:
        # 创建目标文件名
        base_filename = os.path.splitext(os.path.basename(parquet_file))[0]
        output_file = os.path.join(output_path, f"{base_filename}_tokens.jsonl")
        
        # 使用PyArrow读取parquet文件的元数据，获取总行数
        parquet_metadata = pq.read_metadata(parquet_file)
        total_rows = parquet_metadata.num_rows
        batch_size = 100
        
        # 使用 mmap 读取 parquet 文件
        with open(parquet_file, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            # 使用 io.BytesIO 将 mmap 对象包装成文件对象
            buffer = io.BytesIO(mm)
            
            pf = pq.ParquetFile(buffer)  # 使用 mmap 包装的 buffer
        
            progress = tqdm(total=total_rows,
                           desc=f"Processing {os.path.basename(parquet_file)}", 
                           position=multiprocessing.current_process()._identity[0] % 10)
            
            current_row = 0
            idx = 0
            for batch in pf.iter_batches(batch_size=batch_size):
                df_batch = batch.to_pandas()
                
                # 处理当前批次中的每一行
                for _, row in df_batch.iterrows():
                    current_row += 1
                    audio_obj = row['audio']
                    audio_data = audio_obj['bytes']
                    transcription = row['transcription']
                    language = row['language']
                    speaker = row['speaker']
                    if speaker not in results:
                        results[speaker] = {}
                    if language not in results[speaker]:
                        results[speaker][language] = []
                    if len(results[speaker][language]) >= 10:
                        progress.update(1)
                        continue
                    
                    with io.BytesIO(audio_data) as audio_buffer:
                        prompt_data, sample_rate = sf.read(audio_buffer)
                        # 确保是单声道，并转换为float32
                        if len(prompt_data.shape) > 1:
                            prompt_data = prompt_data[:, 0]
                        prompt_data = prompt_data.astype(np.float32)
                                
                        # 重采样到16kHz (如果需要)
                        if sample_rate != 16000:
                            prompt_data = resample(prompt_data, orig_sr=sample_rate, target_sr=16000)
                                
                        prompt_speech_16k = torch.tensor(prompt_data).unsqueeze(0)

                        feat = kaldi.fbank(prompt_speech_16k,
                            num_mel_bins=80,
                            dither=0,
                            sample_frequency=16000)
                        feat = feat - feat.mean(dim=0,keepdim=True)
                        embedding = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
                        
                        results[speaker][language].append(embedding)
           
                    progress.update(1)
            
            # 关闭 mmap 对象
            mm.close()
 
 
 
        print(f'All speakers {results.keys()}')
        for speaker in results:
            print(f'{speaker} : All languages {results[speaker].keys()} in {os.getpid()}')
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error processing {parquet_file}: {str(e)}"
def process_file_x(file_info):
    """处理单个parquet文件的函数，每个进程调用一次"""
    parquet_file, output_path, speaker_extractor, device = file_info
    
    # 为每个进程创建独立的speech_tokenizer_session
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    ort_session = InferenceSession(speaker_extractor, sess_options=option,
                                               providers=["CPUExecutionProvider"])
    results = {}
    try:
        # 创建目标文件名
        base_filename = os.path.splitext(os.path.basename(parquet_file))[0]
        output_file = os.path.join(output_path, f"{base_filename}_tokens.jsonl")
        
        # 使用PyArrow读取parquet文件的元数据，获取总行数
        parquet_metadata = pq.read_metadata(parquet_file)
        total_rows = parquet_metadata.num_rows
        batch_size = 100
        
        pf = pq.ParquetFile(parquet_file)
        
        progress = tqdm(total=total_rows,
                       desc=f"Processing {os.path.basename(parquet_file)}", 
                       position=multiprocessing.current_process()._identity[0] % 10)
        
        current_row = 0
        idx = 0
        for batch in pf.iter_batches(batch_size=batch_size):
            df_batch = batch.to_pandas()
            
            # 处理当前批次中的每一行
            for _, row in df_batch.iterrows():
                current_row += 1
                audio_obj = row['audio']
                audio_data = audio_obj['bytes']
                transcription = row['transcription']
                language = row['language']
                speaker = row['speaker']
                if speaker not in results:
                    results[speaker] = {}
                if language not in results[speaker]:
                    results[speaker][language] = []
                if len(results[speaker][language]) >= 10:
                    progress.update(1)
                    continue
                
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

                    feat = kaldi.fbank(prompt_speech_16k,
                        num_mel_bins=80,
                        dither=0,
                        sample_frequency=16000)
                    feat = feat - feat.mean(dim=0,keepdim=True)
                    embedding = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
                    
                    results[speaker][language].append(embedding)
       
                progress.update(1)
            
 
 
 
 
        print(f'All speakers {results.keys()}')
        for speaker in results:
            print(f'{speaker} : All languages {results[speaker].keys()} in {os.getpid()}')
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error processing {parquet_file}: {str(e)}"
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/external_data/yueyudata/starrail-voice')
    parser.add_argument('--output_path',type=str,default='/external_data/yueyudata/starrail-voice-speaker-embeddings')
    parser.add_argument('--speaker_extractor',type=str,default='/external_data/models/CosyVoice2-0.5B_RWKV_1.5B/campplus.onnx')
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument('--num_processes',type=int,default=4)
    args = parser.parse_args()
    
    print(args)
    data_path = args.data_path
    output_path = args.output_path
    device = args.device
    speaker_extractor = args.speaker_extractor
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
    file_info_list = [(file, output_path, speaker_extractor, device) for file in parquet_files]
    
    # 使用进程池处理文件
    print(f"Starting processing with {num_processes} processes")
    
    # 使用进程池处理文件
    print(f"Starting processing with {num_processes} processes")
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_file, file_info_list)
    
    # 输出处理结果
    print('Processing complete,merge results')
    final_results = {}
    for result in results:
        if isinstance(result, dict):
            for speaker in result:
                if speaker not in final_results:
                    final_results[speaker] = {}
                for language in result[speaker]:
                    if language not in final_results[speaker]:
                        final_results[speaker][language] = []
                    final_results[speaker][language].extend(result[speaker][language])
        else:
            print(result)
    
    # 输出结果
    for speaker in final_results:
        for language in final_results[speaker]:
            output_file = os.path.join(output_path, f"{speaker}_{language}_embeddings.json")
            print(f"Writing embeddings for {speaker} ({language}) to {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f_out:
                json.dump(final_results[speaker][language], f_out)