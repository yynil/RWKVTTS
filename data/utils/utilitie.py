from concurrent.futures import thread
from operator import is_
from librosa import ex
from regex import P
from torch import device
from tqdm import tqdm
import tarfile
import random
import time
import io
import torchaudio
import json
import os
import multiprocessing
import torch
from data.cosy.data.data_processor import init_process, preprocess_prompts
import random

def load_file_list(tar_file):
    #the files are FILE_NAME.mp3/FILE_NAME.json
    #return all FILE_NAME as a list which has a mp3 and json
    import tarfile
    with tarfile.open(tar_file, 'r') as f:
        file_names = f.getnames()
    mp3_files = [i for i in file_names if i.endswith('.mp3')]
    json_files = [i for i in file_names if i.endswith('.json')]
    
    #filter mp3_files without corresponded json
    mp3_files = [i for i in mp3_files if i.replace('.mp3', '.json') in json_files] 
    return mp3_files

def extract_prompt(input_tar_files, input_tar_languages, max_duration=5, num_samples=10, target_sr=16000, output_dir=None):
    """
    Extract prompt from tar files
    Args:
        input_tar_files: list of str, input tar files
        input_tar_languages: list of str, input tar languages for each tar file, must be the same length as input_tar_files
        max_duration: float, max duration of audio
        num_samples: int, number of samples to extract
        target_sr: int, target sample rate
        output_dir: str, output directory
    """
    for tar_file, language in zip(input_tar_files, input_tar_languages):
        print(f'Extracting prompt from {tar_file}...with language {language}')
        random.seed(time.time())
        samples = []
        mp3_files = load_file_list(tar_file)
        with tarfile.open(tar_file, 'r') as f:
            progress_bar = tqdm(total=num_samples,desc=f'Extracting prompt from {tar_file}')
            for i in random.sample(mp3_files, len(mp3_files)):
                mp3 = f.extractfile(i)
                mp3_bytes = io.BytesIO(mp3.read())
                speech, sample_rate = torchaudio.load(mp3_bytes,backend='soundfile')
                json_file = f.extractfile(i.replace('.mp3', '.json'))
                json_data = json.load(json_file)
                duration = json_data['duration']
                if duration > max_duration:
                    continue
                speech = speech.mean(dim=0, keepdim=True)
                if sample_rate != target_sr:
                    assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
                    speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
                samples.append((speech, json_data,sample_rate))
                progress_bar.update(1)
                if len(samples) == num_samples:
                    break
        if output_dir is not None:
            """
            json looks like:
            {'id': 'ZH_B00000_S01450_W000017', 'wav': 'ZH_B00000/ZH_B00000_S01450/mp3/ZH_B00000_S01450_W000017.mp3', 'text': '因此，我们认为流通性具有更广泛的含义。', 'duration': 4.193, 'speaker': 'ZH_B00000_S01450', 'language': 'zh', 'dnsmos': 3.3709}
            """
            output_dir_lang = os.path.join(output_dir, language)
            os.makedirs(output_dir_lang, exist_ok=True)
            progress_bar = tqdm(total=len(samples), desc=f'Saving samples to {output_dir_lang}')
            for i, (speech, json_data, sample_rate) in enumerate(samples):
                id = json_data['id']
                wave_file = os.path.join(output_dir_lang, f'{id}.wav')
                json_file = os.path.join(output_dir_lang, f'{id}.json')
                torchaudio.save(wave_file, speech, target_sr)
                with open(json_file, 'w') as f:
                    json.dump(json_data, f,ensure_ascii=False)
                progress_bar.update(1)
        print(f'Extracted {len(samples)} samples from {tar_file} with language {language}')

frontend = None
llm = None
cosyvoice = None
output_fp = None
prompts = None
global_device = None
processed_count = 0
def initialize_process(model_dir,prompts_dir,output_dir,device):
    current_process = multiprocessing.current_process()
    file_name = f'{output_dir}/{current_process.pid}.jsonl'
    global frontend,llm,cosyvoice,output_fp,prompts,global_device
    global_device = device
    output_fp = open(file_name, 'w')
    print(f'Initializing process with device {device} and output file {file_name}')
    frontend,llm,cosyvoice = init_process(model_dir,device)
    prompts = preprocess_prompts(frontend,prompts_dir)
    print(f'load prompts {prompts.keys()}')
    return frontend,llm,cosyvoice

def generate_speech_tokens(llm,frontend,tts_text,model_input,device):
    tts_text = frontend.text_normalize(tts_text,split=False, text_frontend=True)
    tts_text_token, tts_text_token_len = frontend._extract_text_token(tts_text)
    tts_text_token_len = torch.tensor([tts_text_token.shape[1]], dtype=torch.int32).to(device)
    prompt_text = model_input['prompt_text'].to(device)
    prompt_text_len = torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(device)
    llm_prompt_speech_token = model_input['llm_prompt_speech_token'].to(device)
    prompt_speech_token_len = torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(device)
    flow_prompt_speech_token = model_input['flow_prompt_speech_token'].to(device)
    prompt_speech_feat = model_input['prompt_speech_feat'].to(device)
    llm_embedding = model_input['llm_embedding'].to(device)
    flow_embedding = model_input['flow_embedding'].to(device)
    speech_tokens = []
    with torch.no_grad():
        for i in llm.inference(text = tts_text_token, 
                            text_len = tts_text_token_len, 
                            prompt_text = prompt_text,
                            prompt_text_len = prompt_text_len,
                            prompt_speech_token = llm_prompt_speech_token,
                            prompt_speech_token_len = prompt_speech_token_len,
                            embedding=llm_embedding
                            ):
            speech_tokens.append(i)
    return speech_tokens

def process_text(text,language):
    global frontend,llm,cosyvoice,output_fp,prompts,processed_count,global_device
    processed_count += 1
    if processed_count % 100 == 0:
        print(f'Processed {processed_count} samples')
    tts_text = text
    splits_txt_by_lines = tts_text.split('\n')
    #remove the sentences with length less than 10
    splits_txt_by_lines = [i.strip() for i in splits_txt_by_lines if len(i.strip()) > 10]
    random.seed(time.time())
    model_input,prompt_text = random.choice(prompts[language])
    llm_prompt_speech_token = model_input['llm_prompt_speech_token'].cpu().tolist()
    for tts_text in splits_txt_by_lines:
        tts_speech_tokens = generate_speech_tokens(llm,frontend,tts_text,model_input,cosyvoice.device)
        output_data = {
            'text': tts_text,
            'tts_speech_tokens': tts_speech_tokens,
            'prompt_text': prompt_text,
            'llm_prompt_speech_token': llm_prompt_speech_token[0]
        }
        output_fp.write(json.dumps(output_data,ensure_ascii=False)+'\n')
        output_fp.flush()
    return processed_count
def process_jsonl_file(jsonl_file,language,process_pool):
    print(f'Processing {jsonl_file}...')
    count = 0
    import json
    with open(jsonl_file, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            data = json.loads(line)
            text = data['text']
            count += 1
            future = process_pool.submit(process_text,text,language)
            print(f'processed {future.result()} requests')
    print(f'Processed {count} samples from {jsonl_file}')
    return count        

def process_parquet_file(parquet_file,language,process_pool):
    print(f'Processing {parquet_file}...')
    import pandas as pd
    df = pd.read_parquet(parquet_file)
    count = 0
    for i in range(len(df)):
        text = df.iloc[i]['text']
        count += 1
        future = process_pool.submit(process_text,text,language)
        print(f'processed {future.result()} requests')
    print(f'Processed {count} samples from {parquet_file}')
    return count

def generate_speech_tokens_single_process(cosy_model_dir, prompts_dir, output_dir, language, jsonl_files=None, parquet_files=None, device="cuda:0",is_cross_lingual=False):
    """
    单进程单线程版本的语音标记生成函数
    """
    import torch
    import json
    import os
    import random
    import time
    import traceback
    import logging
    import sys
    from datetime import datetime
    from data.cosy.data.data_processor import init_process, preprocess_prompts
    
    # 设置日志
    output_dir_lang = os.path.join(output_dir, language)
    os.makedirs(output_dir_lang, exist_ok=True)
    process_id = os.getpid()
    log_file = os.path.join(output_dir_lang, f'process_{process_id}_log.txt')
    
    # 配置日志输出到文件和控制台
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(f'process_{process_id}')
    
    # 记录启动信息
    logger.info(f"='='='='='='='='='='='='='='='='='='='='='='='='='='='='='")
    logger.info(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"进程ID: {process_id}")
    logger.info(f"设备: {device}")
    logger.info(f"模型目录: {cosy_model_dir}")
    logger.info(f"提示词目录: {prompts_dir}")
    logger.info(f"输出目录: {output_dir_lang}")
    if jsonl_files:
        logger.info(f"JSONL文件: {jsonl_files}")
    if parquet_files:
        logger.info(f"Parquet文件: {parquet_files}")
    logger.info(f"='='='='='='='='='='='='='='='='='='='='='='='='='='='='='")
    
    output_fp = None
    frontend = None
    llm = None
    cosyvoice = None
    total_processed = 0
    
    try:
        # 初始化模型
        logger.info(f'初始化模型，使用设备: {device}')
        frontend, llm, cosyvoice = init_process(cosy_model_dir, device)
        
        # 预处理提示
        logger.info(f'开始预处理提示词')
        prompts = preprocess_prompts(frontend, prompts_dir)
        logger.info(f'加载提示完成: {prompts.keys()}')
        
        output_file = os.path.join(output_dir_lang, f'{process_id}.jsonl')
        output_fp = open(output_file, 'w')    
        
        # 处理函数
        def process_single_text(text):
            try:
                tts_text = text
                splits_txt_by_lines = tts_text.split('\n')
                # 删除长度小于10的句子
                splits_txt_by_lines = [i.strip() for i in splits_txt_by_lines if len(i.strip()) > 10]
                
                if not splits_txt_by_lines:
                    logger.warning(f"文本没有有效句子: '{text[:100]}...'")
                    return 0
                
                random.seed(time.time())
                cross_linguals_map = {
                    'zh': 'en',
                    'en': 'zh'
                }
                try:
                    model_input, prompt_text = random.choice(prompts[language if not is_cross_lingual else cross_linguals_map[language]])
                except KeyError:
                    logger.error(f"语言 '{language}' 在提示词中不存在! 可用语言: {list(prompts.keys())}")
                    return 0
                
                llm_prompt_speech_token = model_input['llm_prompt_speech_token'].cpu().tolist()
                
                processed_count = 0
                for tts_text in splits_txt_by_lines:
                    try:
                        # 生成语音标记
                        tts_speech_tokens = generate_speech_tokens(llm, frontend, tts_text, model_input, device)
                        output_data = {
                            'text': tts_text,
                            'tts_speech_tokens': tts_speech_tokens,
                            'prompt_text': prompt_text,
                            'llm_prompt_speech_token': llm_prompt_speech_token[0]
                        }
                        output_fp.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                        output_fp.flush()
                        processed_count += 1
                    except Exception as e:
                        logger.error(f"处理单个句子时出错: '{tts_text[:100]}...'")
                        logger.error(f"错误信息: {str(e)}")
                        logger.error(traceback.format_exc())
                
                return processed_count
            except Exception as e:
                logger.error(f"处理文本块时出错")
                logger.error(f"错误信息: {str(e)}")
                logger.error(traceback.format_exc())
                return 0
        
        # 收集要处理的文件
        files_to_process = []
        
        # 处理JSONL文件
        if jsonl_files is not None:
            logger.info(f"处理指定的JSONL文件")
            for file in jsonl_files:
                if file.endswith('.jsonl'):
                    files_to_process.append(('jsonl', file))
            logger.info(f"共有 {len([f for t, f in files_to_process if t == 'jsonl'])} 个JSONL文件需要处理")
        
        # 处理Parquet文件
        if parquet_files is not None:
            logger.info(f"处理指定的Parquet文件")
            for file in parquet_files:
                if file.endswith('.parquet'):
                    files_to_process.append(('parquet', file))
            logger.info(f"共有 {len([f for t, f in files_to_process if t == 'parquet'])} 个Parquet文件需要处理")
        
        # 顺序处理所有文件
        for file_type, file_path in files_to_process:
            logger.info(f'开始处理文件: {file_path}')
            try:
                if file_type == 'jsonl':
                    # 处理JSONL文件
                    # 首先计算文件总行数，用于进度条
                    total_lines = 0
                    with open(file_path, 'r') as f:
                        for line in f:
                            if line.strip():  # 只计算非空行
                                total_lines += 1
                    
                    logger.info(f"JSONL文件 {file_path} 共有 {total_lines} 行")
                    # 使用进度条处理文件
                    with open(file_path, 'r') as f:
                        from tqdm import tqdm
                        progress_bar = tqdm(total=total_lines, desc=f'处理JSONL文件: {os.path.basename(file_path)}')
                        file_processed = 0
                        for line in f:
                            line = line.strip()
                            if len(line) == 0:
                                continue
                            try:
                                data = json.loads(line)
                                text = data['text']
                                processed = process_single_text(text)
                                total_processed += processed
                                file_processed += processed
                                progress_bar.update(1)
                                progress_bar.set_postfix(total=total_processed)
                            except Exception as e:
                                logger.error(f"处理JSONL行时出错: {line[:100]}...")
                                logger.error(f"错误信息: {str(e)}")
                                logger.error(traceback.format_exc())
                        progress_bar.close()
                        logger.info(f"JSONL文件 {file_path} 完成处理，成功处理 {file_processed} 条记录")
                
                elif file_type == 'parquet':
                    # 处理Parquet文件
                    try:
                        import pandas as pd
                        logger.info(f"加载Parquet文件: {file_path}")
                        df = pd.read_parquet(file_path)
                        logger.info(f"Parquet文件 {file_path} 共有 {len(df)} 行")
                        
                        from tqdm import tqdm
                        progress_bar = tqdm(total=len(df), desc=f'处理Parquet文件: {os.path.basename(file_path)}')
                        file_processed = 0
                        for i in range(len(df)):
                            try:
                                text = df.iloc[i]['text']
                                processed = process_single_text(text)
                                total_processed += processed
                                file_processed += processed
                                progress_bar.update(1)
                                progress_bar.set_postfix(total=total_processed)
                            except Exception as e:
                                logger.error(f"处理Parquet行 {i} 时出错")
                                logger.error(f"错误信息: {str(e)}")
                                logger.error(traceback.format_exc())
                        progress_bar.close()
                        logger.info(f"Parquet文件 {file_path} 完成处理，成功处理 {file_processed} 条记录")
                    except ImportError:
                        logger.error("处理Parquet文件需要pandas库，请安装: pip install pandas")
                    except Exception as e:
                        logger.error(f"处理Parquet文件 {file_path} 时出现错误")
                        logger.error(f"错误信息: {str(e)}")
                        logger.error(traceback.format_exc())
            except Exception as e:
                logger.error(f"处理文件 {file_path} 时出现错误")
                logger.error(f"错误信息: {str(e)}")
                logger.error(traceback.format_exc())
        
        logger.info(f'总共成功处理 {total_processed} 个样本，结果保存到 {output_file}')
    
    except Exception as e:
        logger.error("处理过程中出现全局错误")
        logger.error(f"错误信息: {str(e)}")
        logger.error(traceback.format_exc())
    
    finally:
        # 确保资源正确关闭
        logger.info("清理资源...")
        if output_fp is not None:
            try:
                output_fp.close()
                logger.info(f"关闭输出文件")
            except Exception as e:
                logger.error(f"关闭输出文件时出错: {str(e)}")
        
        # 释放GPU资源
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("已清理GPU缓存")
            except Exception as e:
                logger.error(f"清理GPU缓存时出错: {str(e)}")
        
        logger.info(f"处理结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"='='='='='='='='='='='='='='='='='='='='='='='='='='='='='")

if __name__ == '__main__':
    import argparse
    """
    Parse arguments
    task: str, including 'extract_prompt'
    input_tar_files: list of str, input tar files
    input_tar_languages: list of str, input tar languages for each tar file, must be the same length as input_tar_files
    max_duration: float, max duration of audio
    num_samples: int, number of samples to extract
    target_sr: int, target sample rate
    output_dir: str, output directory
    num_processes: int, number of processes to use
    prompt_dir: str, prompt directory which contains prompt jsonl files and audio files
    language: str, language, zh or en
    cosy_model_dir: str, cosy model directory
    device: str, cuda device used to extract speech tokens
    jsonl_files: list of str, jsonl files
    parquet_files: list of str, parquet files
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='task')
    parser.add_argument('--input_tar_files', nargs='+', type=str, help='input tar files')
    parser.add_argument('--input_tar_languages', nargs='+', type=str, help='input tar languages for each tar file')
    parser.add_argument('--output_dir', type=str, help='output directory',required=True)
    parser.add_argument('--max_duration', type=float, default=5, help='max duration of audio')
    parser.add_argument('--num_samples', type=int, default=10, help='number of samples to extract')
    parser.add_argument('--target_sr', type=int, default=16000, help='target sample rate')
    parser.add_argument('--num_processes', type=int, default=1, help='number of processes to use')
    parser.add_argument('--prompts_dir', type=str, help='prompt directory which contains prompt jsonl files and audio files')
    parser.add_argument('--language', type=str, help='language')
    parser.add_argument('--cosy_model_dir', type=str, help='cosy model directory')
    parser.add_argument('--device', type=str, help='cuda device used to extract speech tokens')
    parser.add_argument('--jsonl_files', nargs='+', type=str, help='jsonl files')
    parser.add_argument('--parquet_files', nargs='+', type=str, help='parquet files')
    parser.add_argument('--is_cross_lingual', action='store_true', help='is cross lingual')
    args = parser.parse_args()
    task = args.task
    if task == 'extract_prompt':   
        input_tar_files = args.input_tar_files
        input_tar_languages = args.input_tar_languages
        output_dir = args.output_dir
        assert len(input_tar_files) == len(input_tar_languages), 'input_tar_files and input_tar_languages must have the same length'
        extract_prompt(input_tar_files, input_tar_languages, args.max_duration, args.num_samples, args.target_sr, output_dir)
    elif task == 'generate_speech_tokens':
        prompts_dir = args.prompts_dir
        language = args.language
        cosy_model_dir = args.cosy_model_dir
        jsonl_files = args.jsonl_files
        parquet_files = args.parquet_files
        device = args.device
        is_cross_lingual = args.is_cross_lingual
        # 使用单进程单线程版本替代多进程版本
        generate_speech_tokens_single_process(
            cosy_model_dir=cosy_model_dir,
            prompts_dir=prompts_dir,
            output_dir=args.output_dir,
            language=language,
            jsonl_files=jsonl_files,
            parquet_files=parquet_files,
            device=device,
            is_cross_lingual=is_cross_lingual
        )

