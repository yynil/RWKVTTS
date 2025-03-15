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
from typing import List
import torch
import torchaudio
import io

'''
Natural Language Instruction
Emotion: 高兴(Happy), 悲伤(Sad), 惊讶(Surprised), 愤怒(Angry), 恐惧(Fearful), 厌恶(Disgusted), 冷
静(Calm), 严肃(Serious)
Speaking Rate: 快速(Fast), 非常快速(Very Fast), 慢速(Slow), 非常慢速(Very Slow)
Dialect: 粤语, 四川话, 上海话, 郑州话, 长沙话, 天津话
Role-playing: 神秘(Mysterious), 凶猛(Fierce), 好奇(Curious), 优雅(Elegant), 孤独(Lonely), 机器
人(Robot), 小猪佩奇(Peppa), etc.
Fine-grained Instruction
Vocal Bursts: [laughter], [breath], etc.
Vocal Features: <laughter></laughter>, <strong></strong>
Examples
- 你能用高兴的情感说吗？< |endofprompt| >今天真是太开心了，马上要放假了！I’m so happy,
Spring Festival is coming!
- Please speaking very fast.< |endofprompt| >Today is a happy day, full of laughter and joy.
- 请问你能模仿粤语的口音吗？< |endofprompt| >多保重，早休息。
- 尝试一下以机器人的角色和我交流。< |endofprompt| >接收知识光波！
- [laughter]有时候，看着小孩子们的天真行为[laughter]，我们总会会心一笑。
- She pursued her dreams with <strong>enthusiasm</strong> and <strong>grit</strong>.
'''

emotions = ['高兴', '悲伤', '惊讶', '愤怒', '恐惧', '厌恶', '冷静', '严肃']
emotions_in_english = ['Happy', 'Sad', 'Surprised', 'Angry', 'Fearful', 'Disgusted', 'Calm', 'Serious']
speaking_rates = ['快速', '非常快速', '慢速', '非常慢速']
speaking_rates_in_english = ['Fast', 'Very Fast', 'Slow', 'Very Slow']
dialects = ['普通话','粤语', '四川话', '上海话', '郑州话', '长沙话', '天津话']
dialects_in_english = ['Mandarin','Cantonese', 'Sichuanese', 'Shanghainese', 'Zhengzhou Dialect', 'Changsha Dialect', 'Tianjin Dialect']
role_playings = ['神秘', '凶猛', '好奇', '优雅', '孤独', '机器人', '小猪佩奇']
role_playings_in_english = ['Mysterious', 'Fierce', 'Curious', 'Elegant', 'Lonely', 'Robot', 'Peppa']
vocal_bursts = ['[laughter]', '[breath]']
vocal_features = ['<laughter></laughter>', '<strong></strong>']
end_of_prompt = '<|endofprompt|>'

def generate_in_emotion_in_chinese(text :str):
    templates = [
        '你能用{}的情感说吗？{}{}',
        '请用{}的情感说。{}{}',
        '请用{}的情感表达。{}{}',
        '请用{}的情感说一下。{}{}',
        '请用{}的情感说一句。{}{}'
    ]
    select_emotion = random.choice(emotions)
    return random.choice(templates).format(select_emotion,end_of_prompt,text)

def generate_in_emotion_in_english(text :str):
    templates = [
        'Can you say it with {} emotion?{}{}',
        'Please say it with {} emotion.{}{}',
        'Please express it with {} emotion.{}{}',
        'Please say it with {} emotion.{}{}',
        'Please say a sentence with {} emotion.{}{}'
    ]
    select_emotion = random.choice(emotions_in_english)
    return random.choice(templates).format(select_emotion,end_of_prompt,text)

def generate_speaking_rate_in_chinese(text :str):
    templates = [
        '请用{}的语速说。{}{}',
        '请用{}的语速说一下。{}{}',
        '请用{}的语速说一句。{}{}',
        '请用{}的语速表达。{}{}',   
        '请用{}的语速说。{}{}',
        '请{}地说。{}{}',
        '请{}地说一下。{}{}',
        '请{}地说一句。{}{}',
        '{}的说。{}{}',
        '{}的说一下。{}{}',
        '{}的说一句。{}{}',
        '{}的表达。{}{}'
        
    ]
    select_rate = random.choice(speaking_rates)
    template = random.choice(templates)
    return template.format(select_rate,end_of_prompt,text)

def generate_speaking_rate_in_english(text :str):
    templates = [
        'Please say it with {} speaking rate.{}{}',
        'Say it with {} speaking rate.{}{}',
        'Please say a sentence with {} speaking rate.{}{}',
        'Please express it with {} speaking rate.{}{}',
        'Please speak {}ly.{}{}',
        'Speak {}ly.{}{}',
        'Please say it {}ly.{}{}',
        'Say it {}ly.{}{}'
    ]
    select_rate = random.choice(speaking_rates_in_english)
    template = random.choice(templates)
    return template.format(select_rate,end_of_prompt,text)
        

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

def generate_dialect_in_chinese(text: str):
    templates = [
        '请问你能模仿{}的口音吗？{}{}',
        '请用{}的口音说一下。{}{}',
        '用{}的口音说一句。{}{}',
        '能用{}的口音读一下吗？{}{}',
        '请尝试用{}的口音说这段话。{}{}',
        '请以{}的口音表达。{}{}',
        '请用{}的语调说。{}{}',
        '试试用{}的方言说。{}{}',
        '能否用{}的语调读出来？{}{}',
        '请说一段{}。{}{}'
    ]
    select_dialect = random.choice(dialects)
    return random.choice(templates).format(select_dialect, end_of_prompt, text)

def generate_dialect_in_english(text: str):
    templates = [
        'Can you mimic the {} accent?{}{}',
        'Please speak with a {} accent.{}{}',
        'Say it with a {} accent.{}{}',
        'Could you read this with a {} accent?{}{}',
        'Please try to speak this with a {} accent.{}{}',
        'Please express it with a {} accent.{}{}',
        'Please use {} intonation.{}{}',
        'Try speaking in {}.{}{}',
        'Could you read this in {}?{}{}',
        'Please say a passage in {}.{}{}'
    ]
    select_dialect = random.choice(dialects_in_english)
    return random.choice(templates).format(select_dialect, end_of_prompt, text)

def generate_role_playing_in_chinese(text: str):
    templates = [
        '尝试一下以{}的角色和我交流。{}{}',
        '请以{}的角色说这句话。{}{}',
        '假装你是{}，说一下这句话。{}{}',
        '扮演{}来说这段话。{}{}',
        '请用{}的语气说。{}{}',
        '以{}的形象来表达。{}{}',
        '你能用{}的方式说吗？{}{}',
        '模仿{}说话。{}{}',
        '请用{}的口吻说一下。{}{}',
        '像{}一样说这句话。{}{}'
    ]
    select_role = random.choice(role_playings)
    return random.choice(templates).format(select_role, end_of_prompt, text)

def generate_role_playing_in_english(text: str):
    templates = [
        'Try to communicate with me as a {} character.{}{}',
        'Please say this as a {} character.{}{}',
        'Pretend you are {}, say this sentence.{}{}',
        'Act as {} to say this passage.{}{}',
        'Please speak with a {} tone.{}{}',
        'Express this with a {} image.{}{}',
        'Can you say this in a {} way?{}{}',
        'Mimic {} speaking.{}{}',
        'Please say this in the manner of {}.{}{}',
        'Say this like {}.{}{}'
    ]
    select_role = random.choice(role_playings_in_english)
    return random.choice(templates).format(select_role, end_of_prompt, text)

def generate_vocal_bursts(text: str):
    """
    在文本中随机添加声音爆发标记，如[laughter]、[breath]等
    """
    templates = [
        '{}{}',  # 在句首添加
        '{}{}{}',  # 在句中添加
        '{}{}'  # 在句末添加
    ]
    
    burst = random.choice(vocal_bursts)
    template_choice = random.choice(templates)
    
    if template_choice == '{}{}':  # 句首
        return burst + text
    elif template_choice == '{}{}{}':  # 句中
        words = text.split()
        if len(words) <= 3:  # 文本太短不分割
            return burst + text
        split_point = random.randint(1, len(words) - 1)
        return ' '.join(words[:split_point]) + ' ' + burst + ' ' + ' '.join(words[split_point:])
    else:  # 句末
        return text + ' ' + burst

def generate_vocal_features(text: str):
    """
    在文本中随机添加声音特征标记，如<laughter></laughter>、<strong></strong>等
    支持中文和英文文本
    """
    feature = random.choice(vocal_features)
    feature_start, feature_end = feature.split('><')
    feature_start += '>'
    feature_end = '<' + feature_end
    
    # 检查是否为中文文本
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    
    if has_chinese:
        # 处理中文文本
        if len(text) <= 10:  # 文本太短，整个加强
            return feature_start + text + feature_end
        
        # 对中文处理，随机选择一个字符范围
        text_len = len(text)
        # 随机选择一个起始位置和一个范围长度
        start_pos = random.randint(1, max(1, text_len // 2))  # 避免总是从句首开始
        span_length = random.randint(1, min(5, text_len - start_pos))
        end_pos = start_pos + span_length - 1
        
        # 在选定位置插入标记
        result = text[:start_pos] + feature_start + text[start_pos:end_pos+1] + feature_end + text[end_pos+1:]
        return result
    else:
        # 处理英文文本
        words = text.split()
        if len(words) <= 3:  # 文本太短，整个加强
            return feature_start + text + feature_end
        
        # 随机选择一个词或短语来添加特征
        start_idx = random.randint(0, len(words) - 1)
        span_length = random.randint(1, min(3, len(words) - start_idx))  # 最多3个词
        
        result = []
        for i, word in enumerate(words):
            if i == start_idx:
                result.append(feature_start + word)
            elif i == start_idx + span_length - 1:
                result.append(word + feature_end)
            else:
                result.append(word)
        
        return ' '.join(result)

def generate_mixed_instructions(text: str, language="zh"):
    """
    混合多种指令类型，可以同时包含情感、语速、方言、角色扮演等
    """
    instruction_generators = []
    
    if language == "zh":
        instruction_generators = [
            generate_in_emotion_in_chinese,
            generate_speaking_rate_in_chinese,
            generate_dialect_in_chinese,
            generate_role_playing_in_chinese
        ]
    else:  # 英文
        instruction_generators = [
            generate_in_emotion_in_english,
            generate_speaking_rate_in_english,
            generate_dialect_in_english,
            generate_role_playing_in_english
        ]
    
    # 随机选择1个generator
    selected_generator = random.choice(instruction_generators)
    
    # 可能会添加声音特征
    text_with_features = text
    if random.random() < 0.3:  # 30%的概率添加声音特征
        text_with_features = generate_vocal_features(text)
    
    # 可能会添加声音爆发
    if random.random() < 0.2:  # 20%的概率添加声音爆发
        text_with_features = generate_vocal_bursts(text_with_features)
    
    # 应用选择的指令生成器
    result = text_with_features
    result = selected_generator(result)
    
    return result

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
    prompt_text = model_input['prompt_text'].to(device) if 'prompt_text' in model_input else torch.zeros(1, 0, dtype=torch.int32).to(device)
    prompt_text_len = torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(device) if prompt_text is not None else torch.zeros(1, 0, dtype=torch.int32).to(device)
    llm_prompt_speech_token = model_input['llm_prompt_speech_token'].to(device) if 'llm_prompt_speech_token' in model_input else torch.zeros(1, 0, dtype=torch.int32).to(device)
    prompt_speech_token_len = torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(device) if llm_prompt_speech_token is not None else None
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

def generate_speech_tokens_single_process(cosy_model_dir, prompts_dir, output_dir, language, jsonl_files=None, parquet_files=None, device="cuda:0",is_cross_lingual=False,is_instructed=False):
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
    logger.info(f"='='='='='='='='='='='Instructed={is_instructed}'='='='='='='='='='='='='='='='='='")
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
                
                llm_prompt_speech_token = model_input['llm_prompt_speech_token'].cpu().tolist() if 'llm_prompt_speech_token' in model_input else []
                
                processed_count = 0
                for tts_text in splits_txt_by_lines:
                    try:
                        if is_instructed:
                            tts_text = generate_mixed_instructions(tts_text, language)
                            prompt_text = ""
                            llm_prompt_speech_token[0]=[]
                            if 'prompt_text' in model_input:
                                del model_input['prompt_text']
                            if 'prompt_text_len' in model_input:
                                del model_input['prompt_text_len']
                            if 'llm_prompt_speech_token' in model_input:
                                del model_input['llm_prompt_speech_token']
                            if 'llm_prompt_speech_token_len' in model_input:
                                del model_input['llm_prompt_speech_token_len']
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
    parser.add_argument('--is_instructed', action='store_true', help='is instructed')
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
        is_instructed = args.is_instructed
        # 使用单进程单线程版本替代多进程版本
        generate_speech_tokens_single_process(
            cosy_model_dir=cosy_model_dir,
            prompts_dir=prompts_dir,
            output_dir=args.output_dir,
            language=language,
            jsonl_files=jsonl_files,
            parquet_files=parquet_files,
            device=device,
            is_cross_lingual=is_cross_lingual,
            is_instructed=is_instructed,
        )

