#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RWKV TTS 交互式音频生成工具
使用 webrwkv_py 和 ONNX Runtime 进行音频生成
"""

import os
import sys
import re
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import soundfile as sf
import click

# 抑制警告
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
np.seterr(all='ignore')

# 检查并导入必要的库
try:
    import webrwkv_py
    HAS_WEBRWKV = True
except ImportError:
    HAS_WEBRWKV = False
    print("❌ 错误: 需要安装 'webrwkv_py' 库")
    print("请运行: pip install webrwkv_py")
    sys.exit(1)

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("❌ 错误: 需要安装 'onnxruntime' 库")
    print("请运行: pip install onnxruntime")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("❌ 错误: 需要安装 'transformers' 库")
    print("请运行: pip install transformers")
    sys.exit(1)

try:
    import questionary
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False
    print("❌ 错误: 需要安装 'questionary' 库来使用交互式界面")
    print("请运行: pip install questionary")
    sys.exit(1)

# 导入属性工具
try:
    from utils.properties_util import (
        SPEED_MAP, PITCH_MAP, AGE_MAP, GENDER_MAP, EMOTION_MAP
    )
    # 从映射中提取选项
    age_choices = list(AGE_MAP.keys())
    gender_choices = list(GENDER_MAP.keys())
    emotion_choices = list(EMOTION_MAP.keys())
    pitch_choices = list(PITCH_MAP.keys())
    speed_choices = list(SPEED_MAP.keys())
except ImportError:
    print("⚠️  警告: 无法导入 properties_util，使用默认选项")
    # 默认选项
    age_choices = ['child', 'teenager', 'youth-adult', 'middle-aged', 'elderly']
    gender_choices = ['female', 'male']  # 与properties_util.py保持一致
    emotion_choices = ['NEUTRAL', 'HAPPY', 'SAD', 'ANGRY', 'FEARFUL', 'DISGUSTED', 'SURPRISED']
    pitch_choices = ['low_pitch', 'medium_pitch', 'high_pitch', 'very_high_pitch']
    speed_choices = ['very_slow', 'slow', 'medium', 'fast', 'very_fast']

def detect_token_lang(token: str) -> str:
    """基于字符集合的简单词级语言检测。返回 'en' 或 'zh'。"""
    if not token:
        return 'en'
    has_zh = re.search(r"[\u4e00-\u9fff]", token) is not None
    has_en = re.search(r"[A-Za-z]", token) is not None
    if has_zh and not has_en:
        return 'zh'
    if has_en and not has_zh:
        return 'en'
    if has_zh and has_en:
        return 'zh'
    return 'en'

def sample_logits(logits, temperature=1.0, top_p=0.85, top_k=0):
    """从logits中采样token"""
    if temperature == 0:
        temperature = 1.0
        top_p = 0
    
    if isinstance(logits, list):
        logits = np.array(logits)
    
    try:
        from scipy import special
        probs = special.softmax(logits, axis=-1)
    except ImportError:
        # 如果没有scipy，使用numpy的简单实现
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
    
    top_k = int(top_k)
    
    sorted_ids = np.argsort(probs)
    sorted_probs = probs[sorted_ids][::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    
    cutoff_mask = cumulative_probs >= top_p
    if np.any(cutoff_mask):
        cutoff_idx = np.argmax(cutoff_mask)
        cutoff = float(sorted_probs[cutoff_idx])
        probs[probs < cutoff] = 0
    
    if top_k < len(probs) and top_k > 0:
        probs[sorted_ids[:-top_k]] = 0
    
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)
    
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), size=1, p=probs)
    return int(out[0])

def get_unique_filename(output_dir, text, extension=".wav"):
    """生成唯一的文件名，避免重名"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = text[:3] if len(text) >= 3 else text
    prefix = re.sub(r'[\W\s]', '', prefix).strip()
    
    base_name = prefix
    index = 0
    
    while True:
        if index == 0:
            filename = base_name + extension
        else:
            filename = f"{base_name}_{index}{extension}"
        
        filepath = output_dir / filename
        if not filepath.exists():
            return str(filepath)
        index += 1

class TTSGenerator:
    """TTS生成器类，负责音频生成和统计"""
    
    def __init__(self, runtime, tokenizer, decoder_path, device, model_path):
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.decoder_path = decoder_path
        self.device = device
        self.model_path = model_path
        
        # 初始化 RefAudioUtilities 实例
        print('🎿 开始加载音频编码器模型')
        try:
            audio_tokenizer_path = os.path.join(model_path, 'BiCodecTokenize.onnx')
            wav2vec2_path = os.path.join(model_path, 'wav2vec2-large-xlsr-53.onnx')
            from utils.ref_audio_utilities import RefAudioUtilities
            self.ref_audio_utilities = RefAudioUtilities(audio_tokenizer_path, wav2vec2_path)
            print('✅ 音频编码器模型加载成功')
        except Exception as e:
            print(f'❌ 音频编码器模型加载失败: {e}')
            self.ref_audio_utilities = None
        
        # 缓存ONNX session
        print('🎿 开始加载ONNX模型')
        try:
            self.ort_session = ort.InferenceSession(decoder_path)
            print('✅ ONNX模型加载成功')
        except Exception as e:
            print(f'❌ ONNX模型加载失败: {e}')
            raise
        
        # 生成统计信息
        self.generation_stats = {
            'total_generations': 0,
            'total_tokens': 0,
            'total_time': 0.0,
            'last_generation': {
                'text': '',
                'params': {},
                'total_time': 0.0,
                'total_tokens': 0,
                'audio_duration': 0.0,
                'rtf': 0.0,
                'global_speed': 0.0,
                'semantic_speed': 0.0,
                'decode_speed': 0.0,
                'timestamp': '',
                'output_path': ''
            }
        }
    
    def reset_runtime(self):
        """重置runtime状态"""
        try:
            self.runtime.reset()
            print("🔄 Runtime状态已重置")
        except Exception as e:
            print(f"⚠️  Runtime重置失败: {e}")
    
    def generate_audio(self, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """生成音频"""
        start_time = time.time()
        
        # 重置runtime状态
        self.reset_runtime()
        
        # 获取参数
        text = params['text']
        
        # 检查是否为 zero shot 模式
        if params.get('zero_shot', False):
            # Zero shot 模式
            ref_audio_path = params['ref_audio_path']
            prompt_text = params.get('prompt_text', "希望你以后能够做的，比我还好呦！")
            
            print(f"🎯 开始生成音频 (Zero Shot 模式): {text}")
            print(f"📊 参数: 参考音频={ref_audio_path}, 提示文本={prompt_text}")
            
            # 检测语言
            lang = detect_token_lang(text)
            print(f"🌍 检测到语言: {lang}")
            
            # 使用 zero shot 方法生成 tokens
            global_tokens, semantic_tokens, global_time, global_speed, semantic_time, semantic_speed = self._generate_tokens_zeroshot(text, ref_audio_path, prompt_text)
        else:
            # 传统模式
            age = params['age']
            gender = params['gender']
            emotion = params['emotion']
            pitch = params['pitch']
            speed = params['speed']
            
            print(f"🎯 开始生成音频: {text}")
            print(f"📊 参数: 年龄={age}, 性别={gender}, 情感={emotion}, 音高={pitch}, 速度={speed}")
            
            # 检测语言
            lang = detect_token_lang(text)
            print(f"🌍 检测到语言: {lang}")
            
            # 生成global tokens和semantic tokens
            global_tokens, semantic_tokens, global_time, global_speed, semantic_time, semantic_speed = self._generate_tokens(text, age, gender, emotion, pitch, speed)
        
        # 解码音频
        print("🎵 解码音频...")
        decode_start = time.time()
        
        # 准备输入数据 - 按照tts_gui_simple.py的逻辑
        print("🔧 准备解码器输入数据...")
        global_tokens_array = np.array(global_tokens, dtype=np.int64).reshape(1, 1, -1)
        semantic_tokens_array = np.array(semantic_tokens, dtype=np.int64).reshape(1, -1)
        print(f'🎯 生成的全局token: {global_tokens}')
        print(f'🎯 生成的语义token: {semantic_tokens}')
        print(f'📊 解码器输入形状: global_tokens={global_tokens_array.shape}, semantic_tokens={semantic_tokens_array.shape}')
        
        # 使用ONNX解码器生成音频
        print("🎵 开始ONNX解码器推理...")
        outputs = self.ort_session.run(None, {
                "global_tokens": global_tokens_array, 
                "semantic_tokens": semantic_tokens_array
            })
        wav_data = outputs[0].reshape(-1)
        decode_time = time.time() - decode_start
        
        # 计算音频时长和RTF
        audio_duration = len(wav_data) / 16000  # 采样率16kHz
        decode_speed = len(semantic_tokens) / decode_time if decode_time > 0 else 0
        total_time = time.time() - start_time
        total_tokens = len(global_tokens) + len(semantic_tokens)
        rtf = total_time / audio_duration if audio_duration > 0 else 0
        
        print(f"✅ 音频解码完成，时长 {audio_duration:.2f}s，耗时 {decode_time:.2f}s，速度 {decode_speed:.1f} tokens/s")
        print(f"📊 总耗时: {total_time:.2f}s，RTF: {rtf:.2f}")
        
        # 更新统计信息
        self.generation_stats['total_generations'] += 1
        self.generation_stats['total_tokens'] += total_tokens
        self.generation_stats['total_time'] += total_time
        
        self.generation_stats['last_generation'] = {
            'text': text,
            'params': params,
            'total_time': total_time,
            'total_tokens': total_tokens,
            'audio_duration': audio_duration,
            'rtf': rtf,
            'global_speed': global_speed,
            'semantic_speed': semantic_speed,
            'decode_speed': decode_speed,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'output_path': ''
        }
        
        return wav_data, self.generation_stats['last_generation']
    
    def _generate_tokens(self, text: str, age: str, gender: str, emotion: str, pitch: str, speed: str) -> Tuple[List[int], List[int], float, float, float, float]:
        """
        生成global tokens和semantic tokens
        
        Args:
            text: 原始文本内容
            age: 年龄参数
            gender: 性别参数
            emotion: 情感参数
            pitch: 音高参数
            speed: 速度参数
            
        Returns:
            Tuple: (global_tokens, semantic_tokens, global_time, global_speed, semantic_time, semantic_speed)
        """
        # 编码文本
        print("🔤 编码文本...")
        tokens = self.tokenizer.encode(text)
        print(f"✅ 文本编码完成，共 {len(tokens)} 个token")
        
        # 生成全局token
        print("🌐 生成全局token...")
        global_start = time.time()
        
        # 准备输入tokens
        TTS_TAG_0 = 8193
        TTS_TAG_1 = 8194
        TTS_TAG_2 = 8195
        
        # 构建属性tokens - 使用properties_util.py
        from utils.properties_util import convert_standard_properties_to_tokens
        properties_text = convert_standard_properties_to_tokens(age, gender, emotion, pitch, speed)
        print(f'🔤 属性文本: {properties_text}')
        properties_tokens = self.tokenizer.encode(properties_text, add_special_tokens=False)
        properties_tokens = [i + 8196 + 4096 for i in properties_tokens]
        
        # 构建文本tokens
        text_tokens = [i + 8196 + 4096 for i in tokens]
        
        # 组合所有tokens
        all_idx = properties_tokens + [TTS_TAG_2] + text_tokens + [TTS_TAG_0]
        print(f'🔢 属性token: {properties_tokens}')
        print(f'🔢 文本token: {text_tokens}')
        print(f'🎯 组合后的tokens: {all_idx}')
        
        # Prefill阶段
        print("💎 开始Prefill阶段...")
        logits = self.runtime.predict(all_idx)
        print(f"✅ Prefill完成，logits长度: {len(logits)}")
        
        # 生成全局token - 按照tts_gui_simple.py的逻辑
        print("🌍 开始生成全局token...")
        global_tokens_size = 32
        global_tokens = []
        
        for i in range(global_tokens_size):
            # 从logits中采样token
            sampled_id = sample_logits(logits[0:4096], temperature=1.0, top_p=0.95, top_k=20)
            global_tokens.append(sampled_id)
            # 预测下一个token
            sampled_id += 8196
            logits = self.runtime.predict_next(sampled_id)
        
        global_time = time.time() - global_start
        global_speed = global_tokens_size / global_time if global_time > 0 else 0
        print(f"✅ 全局token生成完成，共 {len(global_tokens)} 个token，耗时 {global_time:.2f}s，速度 {global_speed:.1f} tokens/s")
        print(f'🎯 生成的全局token: {global_tokens}')
        
        # 生成语义token
        print("🧠 生成语义token...")
        semantic_start = time.time()
        
        # 按照tts_gui_simple.py的逻辑生成语义token
        x = self.runtime.predict_next(TTS_TAG_1)
        semantic_tokens = []
        
        for i in range(2048):  # 最大生成2048个token
            sampled_id = sample_logits(x[0:8193], temperature=1.0, top_p=0.95, top_k=80)
            if sampled_id == 8192:  # 遇到结束标记
                print(f"🛑 语义token生成结束，遇到结束标记，共生成 {len(semantic_tokens)} 个token")
                break
            semantic_tokens.append(sampled_id)
            x = self.runtime.predict_next(sampled_id)
        
        semantic_time = time.time() - semantic_start
        semantic_speed = len(semantic_tokens) / semantic_time if semantic_time > 0 else 0
        print(f"✅ 语义token生成完成，共 {len(semantic_tokens)} 个token，耗时 {semantic_time:.2f}s，速度 {semantic_speed:.1f} tokens/s")
        
        return global_tokens, semantic_tokens, global_time, global_speed, semantic_time, semantic_speed

    def _generate_tokens_zeroshot(self, text: str, ref_audio_path: str, prompt_text: str = "希望你以后能够做的，比我还好呦！") -> Tuple[List[int], List[int], float, float, float, float]:
        """
        使用 zero shot 方式生成global tokens和semantic tokens
        
        Args:
            text: 原始文本内容
            ref_audio_path: 参考音频路径
            prompt_text: 提示文本，默认为"希望你以后能够做的，比我还好呦！"
            
        Returns:
            Tuple: (global_tokens, semantic_tokens, global_time, global_speed, semantic_time, semantic_speed)
        """
        if self.ref_audio_utilities is None:
            raise RuntimeError("RefAudioUtilities 未初始化，无法使用 zero shot 模式")
        
        # 编码文本
        print("🔤 编码文本...")
        text_tokens = self.tokenizer.encode(prompt_text + text, add_special_tokens=False)
        text_tokens = [i + 8196 + 4096 for i in text_tokens]
        print(f"✅ 文本编码完成，共 {len(text_tokens)} 个token")
        
        # 从参考音频获取 global tokens 和 semantic tokens
        print("🎵 处理参考音频...")
        global_tokens, prompt_semantic_tokens = self.ref_audio_utilities.tokenize(ref_audio_path)
        print(f"✅ 参考音频处理完成")
        
        # 直接使用flatten()展平数组并转换为Python一维数组
        global_tokens = [int(i) + 8196 for i in global_tokens.flatten()]
        prompt_semantic_tokens = [int(i) for i in prompt_semantic_tokens.flatten()]
        
        print(f'🎯 参考音频 global_tokens: {global_tokens}')
        print(f'🎯 参考音频 semantic_tokens: {prompt_semantic_tokens}')
        
        # 生成全局token
        print("🌐 生成全局token...")
        global_start = time.time()
        
        # 准备输入tokens
        TTS_TAG_0 = 8193
        TTS_TAG_1 = 8194
        TTS_TAG_2 = 8195
        
        # 组合所有tokens
        all_idx = [TTS_TAG_2] + text_tokens + [TTS_TAG_0] + global_tokens + [TTS_TAG_1] + prompt_semantic_tokens
        print(f'🎯 组合后的tokens: {all_idx}')
        
        # Prefill阶段
        print("💎 开始Prefill阶段...")
        logits = self.runtime.predict(all_idx)
        print(f"✅ Prefill完成，logits长度: {len(logits)}")
        
        global_time = time.time() - global_start
        global_speed = len(global_tokens) / global_time if global_time > 0 else 0
        print(f"✅ 全局token处理完成，共 {len(global_tokens)} 个token，耗时 {global_time:.2f}s，速度 {global_speed:.1f} tokens/s")
        
        # 生成语义token
        print("🧠 生成语义token...")
        semantic_start = time.time()
        
        # 从当前logits开始生成语义token
        x = logits
        semantic_tokens = []
        
        for i in range(2048):  # 最大生成2048个token
            sampled_id = sample_logits(x[0:8193], temperature=1.0, top_p=0.95, top_k=80)
            if sampled_id == 8192:  # 遇到结束标记
                print(f"🛑 语义token生成结束，遇到结束标记，共生成 {len(semantic_tokens)} 个token")
                break
            semantic_tokens.append(sampled_id)
            x = self.runtime.predict_next(sampled_id)
        
        semantic_time = time.time() - semantic_start
        semantic_speed = len(semantic_tokens) / semantic_time if semantic_time > 0 else 0
        print(f"✅ 语义token生成完成，共 {len(semantic_tokens)} 个token，耗时 {semantic_time:.2f}s，速度 {semantic_speed:.1f} tokens/s")
        
        global_tokens = [i - 8196 for i in global_tokens]
        return global_tokens, semantic_tokens, global_time, global_speed, semantic_time, semantic_speed

def display_stats(stats: Dict[str, Any]):
    """显示生成统计信息"""
    print("\n" + "="*60)
    print("📊 生成统计信息")
    print("="*60)
    
    if stats['text']:
        print(f"📝 文本: {stats['text']}")
        print(f"⏱️  总耗时: {stats['total_time']:.2f}s")
        print(f"🎵 音频时长: {stats['audio_duration']:.2f}s")
        print(f"📈 RTF: {stats['rtf']:.2f}")
        print(f"🔢 总token数: {stats['total_tokens']}")
        print(f"🌐 全局token速度: {stats['global_speed']:.1f} tokens/s")
        print(f"🧠 语义token速度: {stats['semantic_speed']:.1f} tokens/s")
        print(f"🎵 解码速度: {stats['decode_speed']:.1f} tokens/s")
        print(f"🕐 时间: {stats['timestamp']}")
        if stats['output_path']:
            print(f"💾 保存路径: {stats['output_path']}")
    else:
        print("暂无生成记录")
    
    print("="*60)

def interactive_parameter_selection(generator: TTSGenerator):
    """交互式参数选择界面"""
    print("\n🎮 进入交互式配置界面")
    print("💡 使用方向键选择，回车确认，Ctrl+C退出")
    
    while True:
        try:
            print("\n" + "="*60)
            print("🎵 RWKV TTS 参数配置")
            print("="*60)
            
            # 选择生成模式
            generation_mode = questionary.select(
                "🎯 请选择生成模式:",
                choices=[
                    "传统模式 (使用属性参数)",
                    "Zero Shot 模式 (使用参考音频)"
                ],
                default="传统模式 (使用属性参数)"
            ).ask()
            
            if generation_mode is None:  # 用户按Ctrl+C
                break
            
            is_zero_shot = generation_mode == "Zero Shot 模式 (使用参考音频)"
            
            # 文本输入
            text = questionary.text(
                "📝 请输入要转换的文本:",
                default=generator.generation_stats['last_generation'].get('text', '你好，世界！')
            ).ask()
            
            if text is None:  # 用户按Ctrl+C
                break
            
            # 输出目录
            output_dir = questionary.text(
                "📁 请输入输出目录:",
                default="./generated_audio"
            ).ask()
            
            if output_dir is None:
                break
            
            if is_zero_shot:
                # Zero Shot 模式参数
                ref_audio_path = questionary.text(
                    "🎵 请输入参考音频路径:",
                    default="zero_shot_prompt.wav"
                ).ask()
                
                if ref_audio_path is None:
                    break
                
                prompt_text = questionary.text(
                    "💬 请输入提示文本 (可选，回车使用默认值):",
                    default="希望你以后能够做的，能比我还好呦！"
                ).ask()
                
                if prompt_text is None:
                    break
                
    
                
                # 确认生成
                confirm = questionary.confirm(
                    f"🚀 确认生成音频 (Zero Shot 模式)?\n"
                    f"文本: {text}\n"
                    f"参考音频: {ref_audio_path}\n"
                    f"提示文本: {prompt_text}\n"
                    f"输出目录: {output_dir}",
                    default=True
                ).ask()
                
                if confirm:
                    # 准备参数
                    params = {
                        'text': text,
                        'zero_shot': True,
                        'ref_audio_path': ref_audio_path,
                        'prompt_text': prompt_text,
                        'output_dir': output_dir
                    }
                    
                    # 生成音频
                    try:
                        wav_data, stats = generator.generate_audio(params)
                        
                        # 生成唯一文件名
                        output_path = get_unique_filename(output_dir, text)
                        
                        # 保存音频
                        sf.write(output_path, wav_data, 16000)
                        stats['output_path'] = output_path
                        
                        print(f"✅ 音频生成成功，保存至: {output_path}")
                        
                        # 显示统计信息
                        display_stats(stats)
                        
                    except Exception as e:
                        print(f"❌ 生成失败: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                # 传统模式参数
                # 年龄选择
                age = questionary.select(
                    "👶 请选择年龄:",
                    choices=age_choices,
                    default=age_choices[3]  # middle-aged
                ).ask()
                
                if age is None:
                    break
                
                # 性别选择
                gender = questionary.select(
                    "👤 请选择性别:",
                    choices=gender_choices,
                    default=gender_choices[0]  # female (第一个选项)
                ).ask()
                
                if gender is None:
                    break
                
                # 情感选择
                emotion = questionary.select(
                    "😊 请选择情感:",
                    choices=emotion_choices,
                    default=emotion_choices[1]  # NEUTRAL
                ).ask()
                
                if emotion is None:
                    break
                
                # 音高选择
                pitch = questionary.select(
                    "🎵 请选择音高:",
                    choices=pitch_choices,
                    default=pitch_choices[1]  # medium_pitch
                ).ask()
                
                if pitch is None:
                    break
                
                # 速度选择
                speed = questionary.select(
                    "⚡ 请选择速度:",
                    choices=speed_choices,
                    default=speed_choices[2]  # medium
                ).ask()
                
                if speed is None:
                    break
                
             
                # 确认生成
                confirm = questionary.confirm(
                    f"🚀 确认生成音频?\n"
                    f"文本: {text}\n"
                    f"参数: 年龄={age}, 性别={gender}, 情感={emotion}, 音高={pitch}, 速度={speed}\n"
                    f"输出目录: {output_dir}",
                    default=True
                ).ask()
                
                if confirm:
                    # 准备参数
                    params = {
                        'text': text,
                        'zero_shot': False,
                        'age': age,
                        'gender': gender,
                        'emotion': emotion,
                        'pitch': pitch,
                        'speed': speed,
                        'output_dir': output_dir
                    }
                    
                    # 生成音频
                    try:
                        wav_data, stats = generator.generate_audio(params)
                        
                        # 生成唯一文件名
                        output_path = get_unique_filename(output_dir, text)
                        
                        # 保存音频
                        sf.write(output_path, wav_data, 16000)
                        stats['output_path'] = output_path
                        
                        print(f"✅ 音频生成成功，保存至: {output_path}")
                        
                        # 显示统计信息
                        display_stats(stats)
                        
                    except Exception as e:
                        print(f"❌ 生成失败: {e}")
                        import traceback
                        traceback.print_exc()
            
            # 询问是否继续
            continue_generation = questionary.confirm(
                "🔄 是否继续生成音频?",
                default=True
            ).ask()
            
            if not continue_generation:
                break
                
        except KeyboardInterrupt:
            print("\n👋 用户中断，退出程序")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("👋 感谢使用 RWKV TTS!")

@click.command()
@click.option('--model_path', required=True, help='RWKV模型路径')
def main(model_path):
    """RWKV TTS 主程序"""
    print("🚀 欢迎使用 RWKV TTS 交互式音频生成工具!")
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型路径不存在: {model_path}")
        return
    
    # 自动构建解码器路径
    decoder_path = os.path.join(model_path, "BiCodecDetokenize.onnx")
    print(f"🔍 自动设置解码器路径: {decoder_path}")
    
    # 检查模型目录中的文件
    print(f"🔍 检查模型目录: {model_path}")
    try:
        model_files = os.listdir(model_path)
        print(f"📁 模型目录中的文件:")
        for file in model_files:
            file_path = os.path.join(model_path, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"   📄 {file} ({size:,} bytes)")
            else:
                print(f"   📁 {file}/")
    except Exception as e:
        print(f"⚠️  无法列出模型目录内容: {e}")
    
    if not os.path.exists(decoder_path):
        print(f"❌ 错误: 解码器路径不存在: {decoder_path}")
        return
    
    # 选择设备
    print("\n💎 选择设备 💎")
    try:
        devices = webrwkv_py.get_available_adapters_py()
    except AttributeError:
        # 如果新API不存在，尝试旧API
        try:
            devices = webrwkv_py.get_available_devices()
        except AttributeError:
            print("❌ 无法获取可用设备列表")
            return
    
    for i, device in enumerate(devices):
        print(f"{i}: {device}")
    
    device_choice = input("请选择设备: ")
    try:
        device_idx = int(device_choice)
        if device_idx < 0 or device_idx >= len(devices):
            print("❌ 无效的设备选择")
            return
        device = devices[device_idx]
        print(f"✅ 选择设备: {device}")
    except ValueError:
        print("❌ 无效的设备选择")
        return
    
    # 加载模型
    print("\n💎 加载模型 💎")
    try:
        # 尝试多种可能的模型文件名
        possible_model_files = [
            'webrwkv.safetensors',
            'model.safetensors',
            'rwkv.safetensors',
            'pytorch_model.bin',
            'model.bin'
        ]
        
        webrwkv_model_path = None
        for model_file in possible_model_files:
            test_path = os.path.join(model_path, model_file)
            if os.path.exists(test_path):
                webrwkv_model_path = test_path
                print(f"✅ 找到模型文件: {model_file}")
                break
        
        if webrwkv_model_path is None:
            print(f"❌ 未找到模型文件")
            print(f"💡 请检查模型目录 {model_path} 中是否包含以下文件之一:")
            for model_file in possible_model_files:
                print(f"   - {model_file}")
            return
        
        print(f"🔍 尝试加载模型文件: {webrwkv_model_path}")
        
        # 尝试新的API
        model = webrwkv_py.Model(webrwkv_model_path, 'fp32', device_idx)
        print(f"✅ 模型加载成功: {webrwkv_model_path}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print(f"💡 请检查:")
        print(f"   1. 模型文件路径是否正确: {webrwkv_model_path}")
        print(f"   2. 模型文件是否完整")
        print(f"   3. 设备索引是否正确: {device_idx}")
        print(f"   4. 模型文件格式是否支持")
        return
    
    # 创建runtime
    print("\n💎 创建 runtime 💎")
    try:
        runtime = model.create_thread_runtime()
        print("✅ runtime 创建成功")
    except Exception as e:
        print(f"❌ runtime 创建失败: {e}")
        return
    
    # 加载tokenizer
    print("\n💎 加载 tokenizer 💎")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"✅ tokenizer 加载成功: {model_path}")
    except Exception as e:
        print(f"❌ tokenizer 加载失败: {e}")
        print(f"💡 请检查模型目录 {model_path} 中是否包含正确的tokenizer文件")
        return
    
    # 创建TTS生成器
    generator = TTSGenerator(runtime, tokenizer, decoder_path, device, model_path)
    
    # 启动交互式界面
    print("\n🎯 启动交互式配置界面...")
    interactive_parameter_selection(generator)

if __name__ == "__main__":
    main()
