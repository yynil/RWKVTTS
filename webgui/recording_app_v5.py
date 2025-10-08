#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RWKV ASR Gradio 5.x 录音应用
使用最新的Gradio 5.x语法
"""

import os
import sys
import time
from pathlib import Path
import click
import torch
import gradio as gr
import numpy as np
import torchaudio
import librosa
from silero_vad import load_silero_vad, get_speech_timestamps

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.asr_inference_with_chatrwkv import (
    load_asr_models, 
    inference_asr, 
    AsrModels
)

class VADProcessor:
    """VAD (Voice Activity Detection) 处理器"""
    
    def __init__(self, device: str = "cpu"):
        """初始化VAD处理器"""
        self.device = device
        self.vad_model = None
        self.sample_rate = 16000
        
    def load_vad_model(self):
        """加载VAD模型"""
        if self.vad_model is None:
            print("正在加载Silero VAD模型...")
            self.vad_model = load_silero_vad()
            print("VAD模型加载完成")
    
    def trim_audio_with_vad(self, audio_path: str, output_path: str = None) -> str:
        """使用VAD修剪音频，去除静音部分"""
        try:
            # 确保VAD模型已加载
            self.load_vad_model()
            
            # 加载音频文件
            waveform, original_sample_rate = torchaudio.load(audio_path)
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 创建重采样器（如果需要）
            resampler = None
            if original_sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(original_sample_rate, self.sample_rate)
                # 重采样到16kHz（VAD模型要求）
                waveform_for_vad = resampler(waveform)
            else:
                waveform_for_vad = waveform
            
            # 获取语音时间戳
            speech_timestamps = get_speech_timestamps(
                waveform_for_vad.squeeze(), 
                self.vad_model, 
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=250,
                min_silence_duration_ms=100,
                window_size_samples=512,
                speech_pad_ms=30
            )
            
            if not speech_timestamps:
                print("未检测到语音活动")
                return audio_path  # 返回原文件
            
            # 计算语音片段的开始和结束时间（在16kHz下）
            start_sample = speech_timestamps[0]['start']
            end_sample = speech_timestamps[-1]['end']
            
            # 如果原始采样率不是16kHz，需要将时间戳转换回原始采样率
            if original_sample_rate != self.sample_rate:
                # 计算采样率比例
                ratio = original_sample_rate / self.sample_rate
                start_sample = int(start_sample * ratio)
                end_sample = int(end_sample * ratio)
                # 确保不超出原始音频长度
                end_sample = min(end_sample, waveform.shape[1])
            
            # 提取语音部分（使用原始采样率的音频）
            trimmed_waveform = waveform[:, start_sample:end_sample]
            
            # 保存修剪后的音频
            if output_path is None:
                # 在原文件名后添加_vad_trimmed
                base_path = Path(audio_path)
                output_path = base_path.parent / f"{base_path.stem}_vad_trimmed{base_path.suffix}"
            
            # 使用原始采样率保存
            torchaudio.save(output_path, trimmed_waveform, original_sample_rate)
            
            print(f"VAD修剪完成: {audio_path} -> {output_path}")
            print(f"原始长度: {len(waveform[0])/original_sample_rate:.2f}秒")
            print(f"修剪后长度: {len(trimmed_waveform[0])/original_sample_rate:.2f}秒")
            print(f"原始采样率: {original_sample_rate}Hz")
            print(f"修剪后采样率: {original_sample_rate}Hz")
            
            return str(output_path)
            
        except Exception as e:
            print(f"VAD处理失败: {str(e)}")
            return audio_path  # 返回原文件
    
    def restore_original_audio(self, audio_path: str) -> str:
        """恢复原始音频（如果存在VAD修剪版本）"""
        try:
            base_path = Path(audio_path)
            if "_vad_trimmed" in base_path.name:
                # 如果当前是修剪版本，尝试找到原始版本
                original_name = base_path.name.replace("_vad_trimmed", "")
                original_path = base_path.parent / original_name
                if original_path.exists():
                    return str(original_path)
            return audio_path
        except Exception as e:
            print(f"恢复原始音频失败: {str(e)}")
            return audio_path
    
    def preprocess_uploaded_audio(self, audio_path: str, output_path: str = None) -> str:
        """预处理上传的音频文件，转换为单声道16kHz"""
        try:
            # 使用librosa加载音频
            audio_data, original_sr = librosa.load(audio_path, sr=None, mono=False)
            
            # 如果是多声道，转换为单声道
            if len(audio_data.shape) > 1:
                audio_data = librosa.to_mono(audio_data)
            
            # 重采样到16kHz
            target_sr = 16000
            if original_sr != target_sr:
                audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
            
            # 保存预处理后的音频到临时目录
            if output_path is None:
                import tempfile
                import uuid
                temp_dir = Path(tempfile.gettempdir()) / "rwkv_asr_preprocessed"
                temp_dir.mkdir(exist_ok=True)
                
                base_path = Path(audio_path)
                unique_id = str(uuid.uuid4())[:8]
                output_path = temp_dir / f"{base_path.stem}_preprocessed_{unique_id}{base_path.suffix}"
            
            # 使用librosa保存音频
            librosa.output.write_wav(output_path, audio_data, target_sr)
            
            print(f"音频预处理完成: {audio_path} -> {output_path}")
            print(f"原始采样率: {original_sr}Hz -> 目标采样率: {target_sr}Hz")
            print(f"音频长度: {len(audio_data)/target_sr:.2f}秒")
            
            return str(output_path)
            
        except Exception as e:
            print(f"音频预处理失败: {str(e)}")
            return audio_path  # 返回原文件
    
    def get_audio_info(self, audio_path: str) -> dict:
        """获取音频文件信息"""
        try:
            # 使用librosa获取音频信息
            audio_data, sample_rate = librosa.load(audio_path, sr=None, mono=False)
            
            # 计算时长
            duration = len(audio_data[0]) / sample_rate if len(audio_data.shape) > 1 else len(audio_data) / sample_rate
            
            # 获取声道数
            channels = audio_data.shape[0] if len(audio_data.shape) > 1 else 1
            
            return {
                "sample_rate": sample_rate,
                "duration": duration,
                "channels": channels,
                "duration_str": f"{duration:.2f}秒",
                "info_str": f"{sample_rate}Hz, {duration:.2f}秒, {channels}声道"
            }
            
        except Exception as e:
            print(f"获取音频信息失败: {str(e)}")
            return {
                "sample_rate": 0,
                "duration": 0,
                "channels": 0,
                "duration_str": "未知",
                "info_str": "信息获取失败"
            }

class Gradio5ASR:
    def __init__(self, audio_lm_path: str, llm_path: str, whisper_path: str, 
                 tokenizer_path: str, device: str, dtype: str):
        """初始化Gradio 5.x ASR应用"""
        self.audio_lm_path = audio_lm_path
        self.llm_path = llm_path
        self.whisper_path = whisper_path
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.dtype = dtype
        
        # 数据类型映射
        dtype_map = {
            'float16': torch.float16,
            'float32': torch.float32,
            'bfloat16': torch.bfloat16
        }
        self.torch_dtype = dtype_map[dtype]
        
        # 模型实例
        self.models: AsrModels = None
        self._models_loading = False  # 模型加载状态标志
        
        # VAD处理器
        self.vad_processor = VADProcessor(device=device)
        
        # 预设音频管理
        self.demo_voices_dir = Path(__file__).parent / "demo_voices"
        self.demo_audio_files = self._load_demo_audio_files()
        
        # 选择历史跟踪
        self.selection_history = set()  # 跟踪所有选择过的音频
        self.current_selection = None   # 当前选中的音频
        
        # 创建界面
        self.interface = self.create_interface()
    
    def _load_demo_audio_files(self):
        """加载预设音频文件列表"""
        demo_files = {"chinese": [], "english": []}
        
        try:
            # 加载中文音频文件
            zh_dir = self.demo_voices_dir / "zh"
            if zh_dir.exists():
                for file_path in zh_dir.glob("*.wav"):
                    demo_files["chinese"].append({
                        "name": file_path.stem,
                        "path": str(file_path),
                        "display_name": f"中文 - {file_path.stem}"
                    })
            
            # 加载英文音频文件
            en_dir = self.demo_voices_dir / "en"
            if en_dir.exists():
                for file_path in en_dir.glob("*.wav"):
                    demo_files["english"].append({
                        "name": file_path.stem,
                        "path": str(file_path),
                        "display_name": f"英文 - {file_path.stem}"
                    })
            
            print(f"加载预设音频文件: 中文 {len(demo_files['chinese'])} 个, 英文 {len(demo_files['english'])} 个")
            return demo_files
            
        except Exception as e:
            print(f"加载预设音频文件失败: {str(e)}")
            return {"chinese": [], "english": []}
    
    def _get_initial_demo_table_data(self):
        """获取初始预设音频表格数据"""
        try:
            # 构建表格数据
            table_data = []
            for lang in ["chinese", "english"]:
                for audio_info in self.demo_audio_files[lang]:
                    # 获取音频信息
                    audio_info_dict = self.vad_processor.get_audio_info(audio_info["path"])
                    
                    # 构建选择按钮文本（使用纯文本和符号）
                    select_button = "🎵 选择"  # 未选择 - 音符
                    
                    table_data.append([
                        audio_info["name"],
                        audio_info_dict["info_str"],
                        f"{'中文' if lang == 'chinese' else '英文'} | {select_button}"
                    ])
            
            print(f"初始化预设音频表格: {len(table_data)} 行")
            return table_data
            
        except Exception as e:
            print(f"初始化预设音频表格失败: {str(e)}")
            return []
    
    def load_models(self):
        """加载ASR模型"""
        if self.models is None and not self._models_loading:
            self._models_loading = True
            print("正在加载ASR模型...")
            try:
                self.models = load_asr_models(
                    self.audio_lm_path, 
                    self.llm_path, 
                    self.whisper_path, 
                    self.tokenizer_path, 
                    self.device, 
                    self.torch_dtype
                )
                print("ASR模型加载完成")
            except Exception as e:
                print(f"ASR模型加载失败: {str(e)}")
                self._models_loading = False
                raise e
            finally:
                self._models_loading = False
        elif self._models_loading:
            print("ASR模型正在加载中，请稍候...")
            # 等待模型加载完成
            while self._models_loading:
                time.sleep(0.1)
    
    def process_audio(self, audio_path: str, language: str, use_vad: bool = False) -> str:
        """处理音频文件"""
        if audio_path is None:
            return "请先录音"
        
        try:
            # 确保模型已加载
            self.load_models()
            
            # 如果启用VAD，先进行音频修剪
            processed_audio_path = audio_path
            if use_vad:
                print("正在使用VAD修剪音频...")
                processed_audio_path = self.vad_processor.trim_audio_with_vad(audio_path)
                print(f"VAD处理完成: {processed_audio_path}")
            
            # 执行ASR推理
            print(f"开始处理音频: {processed_audio_path}, 语言: {language}")
            start_time = time.time()
            
            results = inference_asr(self.models, processed_audio_path, language, self.torch_dtype, self.device)
            decoded_text = self.models.tokenizer.decode(results)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            vad_info = "\n\n(VAD修剪已启用)" if use_vad else ""
            print(f"识别完成，耗时: {processing_time:.2f}秒")
            return f"识别结果: {decoded_text}\n\n处理时间: {processing_time:.2f}秒{vad_info}"
            
        except Exception as e:
            error_msg = f"识别失败: {str(e)}"
            print(error_msg)
            return error_msg
    
    def create_interface(self):
        """创建Gradio 5.x界面"""
        with gr.Blocks(
            title="RWKV ASR 录音识别系统",
            theme=gr.themes.Soft()
        ) as interface:
            
            gr.Markdown("""
            # 🎤 RWKV ASR 录音识别系统
            
            支持中文和英文录音识别，支持录音和上传音频文件。
            """)
            
            # 创建标签页
            with gr.Tabs():
                with gr.Tab("🎙️ 录音识别"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            # 语言选择
                            language = gr.Radio(
                                choices=["chinese", "english"],
                                value="chinese",
                                label="选择识别语言",
                                info="选择语音识别的目标语言"
                            )
                            
                            # VAD选项
                            use_vad = gr.Checkbox(
                                label="启用VAD修剪",
                                value=False,
                                info="自动去除静音部分，提高识别准确性"
                            )
                            
                            # 录音组件 - Gradio 5.x最新语法
                            audio = gr.Audio(
                                sources=["microphone"],
                                type="filepath",
                                label="录音",
                                interactive=True
                            )
                            
                            # 按钮组
                            with gr.Row():
                                recognize_btn = gr.Button("🚀 开始识别", variant="primary")
                                vad_trim_btn = gr.Button("✂️ VAD修剪", variant="secondary")
                                restore_btn = gr.Button("↩️ 恢复原音频", variant="secondary")
                
                with gr.Tab("📁 上传音频"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            # 预设音频表格（放在最上面）
                            demo_audio_table = gr.Dataframe(
                                headers=["文件名", "音频信息", "语言"],
                                datatype=["str", "str", "str"],
                                label="预设音频列表",
                                interactive=True,
                                row_count=10,
                                col_count=3,
                                wrap=True,
                                value=self._get_initial_demo_table_data()  # 启动时自动加载
                            )
                            
                            # 选中的音频路径（隐藏组件）
                            selected_demo_audio = gr.Textbox(
                                visible=False,
                                value=""
                            )
                            
                            # 音频播放组件
                            demo_audio_player = gr.Audio(
                                label="预设音频播放",
                                interactive=False,
                                visible=True
                            )
                            
                            # 分隔线
                            gr.Markdown("---")
                            
                            # 语言选择（上传）
                            language_upload = gr.Radio(
                                choices=["chinese", "english"],
                                value="chinese",
                                label="选择识别语言",
                                info="选择语音识别的目标语言"
                            )
                            
                            # VAD选项（上传）
                            use_vad_upload = gr.Checkbox(
                                label="启用VAD修剪",
                                value=False,
                                info="自动去除静音部分，提高识别准确性"
                            )
                            
                            # 音频上传组件
                            audio_upload = gr.Audio(
                                sources=["upload"],
                                type="filepath",
                                label="上传音频文件",
                                interactive=True
                            )
                            
                            # 预处理按钮
                            preprocess_btn = gr.Button("🔧 预处理音频", variant="secondary")
                            
                            # 按钮组（上传）
                            with gr.Row():
                                recognize_upload_btn = gr.Button("🚀 开始识别", variant="primary")
                                vad_trim_upload_btn = gr.Button("✂️ VAD修剪", variant="secondary")
                                restore_upload_btn = gr.Button("↩️ 恢复原音频", variant="secondary")
                
                
            # 结果显示区域（三个标签页共享）
            with gr.Row():
                with gr.Column(scale=1):
                    # 结果显示
                    output = gr.Textbox(
                        label="识别结果",
                        lines=15,
                        interactive=False,
                        placeholder="识别结果将显示在这里...",
                        show_copy_button=True
                    )
                    
                    # 状态显示
                    status = gr.Textbox(
                        label="状态",
                        lines=2,
                        interactive=False,
                        value="就绪"
                    )
            
            # 处理函数
            def process_recording(audio_path, language, use_vad):
                """处理录音"""
                if audio_path is None:
                    return "请先录音", "等待录音"
                
                status_text = f"正在处理录音... (语言: {language})"
                result = self.process_audio(audio_path, language, use_vad)
                
                if "识别失败" in result:
                    return result, "处理失败"
                else:
                    return result, "处理完成"
            
            def vad_trim_audio(audio_path):
                """VAD修剪音频"""
                if audio_path is None:
                    return "请先录音", "等待录音", None
                
                try:
                    trimmed_path = self.vad_processor.trim_audio_with_vad(audio_path)
                    if trimmed_path != audio_path:
                        return f"VAD修剪完成！\n\n原文件: {audio_path}\n修剪后: {trimmed_path}", "VAD修剪完成", trimmed_path
                    else:
                        return "未检测到语音活动或修剪失败", "VAD处理失败", audio_path
                except Exception as e:
                    return f"VAD修剪失败: {str(e)}", "处理失败", audio_path
            
            def restore_original_audio(audio_path):
                """恢复原始音频"""
                if audio_path is None:
                    return "请先录音", "等待录音", None
                
                try:
                    original_path = self.vad_processor.restore_original_audio(audio_path)
                    if original_path != audio_path:
                        return f"已恢复原始音频！\n\n当前文件: {original_path}", "恢复完成", original_path
                    else:
                        return "未找到原始音频文件", "恢复失败", audio_path
                except Exception as e:
                    return f"恢复失败: {str(e)}", "处理失败", audio_path
            
            # 上传音频处理函数
            def preprocess_uploaded_audio(audio_path):
                """预处理上传的音频"""
                if audio_path is None:
                    return "请先上传音频文件", "等待上传", None
                
                try:
                    preprocessed_path = self.vad_processor.preprocess_uploaded_audio(audio_path)
                    if preprocessed_path != audio_path:
                        return f"音频预处理完成！\n\n原文件: {audio_path}\n预处理后: {preprocessed_path}", "预处理完成", preprocessed_path
                    else:
                        return "预处理失败", "预处理失败", audio_path
                except Exception as e:
                    return f"预处理失败: {str(e)}", "处理失败", audio_path
            
            def process_uploaded_audio(audio_path, language, use_vad, selected_demo_audio=None):
                """处理上传的音频或预设音频"""
                # 优先使用上传的音频，如果没有上传则使用预设音频
                if audio_path and audio_path != "":
                    target_audio = audio_path
                    audio_source = "上传音频"
                elif selected_demo_audio and selected_demo_audio != "":
                    target_audio = selected_demo_audio
                    audio_source = "预设音频"
                else:
                    return "请先上传音频文件或选择预设音频", "等待输入"
                
                status_text = f"正在处理{audio_source}... (语言: {language})"
                result = self.process_audio(target_audio, language, use_vad)
                
                if "识别失败" in result:
                    return result, "处理失败"
                else:
                    return result, "处理完成"
            
            def vad_trim_uploaded_audio(audio_path, selected_demo_audio=None):
                """VAD修剪上传的音频或预设音频"""
                # 优先使用上传的音频，如果没有上传则使用预设音频
                if audio_path and audio_path != "":
                    target_audio = audio_path
                    audio_source = "上传音频"
                elif selected_demo_audio and selected_demo_audio != "":
                    target_audio = selected_demo_audio
                    audio_source = "预设音频"
                else:
                    return "请先上传音频文件或选择预设音频", "等待输入", None
                
                try:
                    trimmed_path = self.vad_processor.trim_audio_with_vad(target_audio)
                    if trimmed_path != target_audio:
                        return f"VAD修剪完成！\n\n原文件: {target_audio}\n修剪后: {trimmed_path}", "VAD修剪完成", trimmed_path
                    else:
                        return "未检测到语音活动或修剪失败", "VAD处理失败", target_audio
                except Exception as e:
                    return f"VAD修剪失败: {str(e)}", "处理失败", target_audio
            
            def restore_uploaded_audio(audio_path, selected_demo_audio=None):
                """恢复原始上传音频或预设音频"""
                # 优先使用上传的音频，如果没有上传则使用预设音频
                if audio_path and audio_path != "":
                    target_audio = audio_path
                    audio_source = "上传音频"
                elif selected_demo_audio and selected_demo_audio != "":
                    target_audio = selected_demo_audio
                    audio_source = "预设音频"
                else:
                    return "请先上传音频文件或选择预设音频", "等待输入", None
                
                try:
                    original_path = self.vad_processor.restore_original_audio(target_audio)
                    if original_path != target_audio:
                        return f"已恢复原始{audio_source}！\n\n当前文件: {original_path}", "恢复完成", original_path
                    else:
                        return f"未找到原始{audio_source}文件", "恢复失败", target_audio
                except Exception as e:
                    return f"恢复失败: {str(e)}", "处理失败", target_audio
            
            # 预设音频处理函数（表格在启动时自动加载，无需手动加载函数）
            
            def on_table_select(evt: gr.SelectData):
                """处理表格选择事件 - 只有点击选择按钮才选择"""
                try:
                    # 获取所有音频文件信息
                    all_files = []
                    all_languages = []
                    for lang in ["chinese", "english"]:
                        for audio_info in self.demo_audio_files[lang]:
                            all_files.append(audio_info["path"])
                            all_languages.append(lang)
                    
                    # 检查是否点击的是选择按钮列（第3列，索引为2）
                    if evt.index[1] == 2:  # 点击的是"选择"按钮列
                        # 检查选择的行是否有效
                        if evt.index[0] < len(all_files):
                            selected_path = all_files[evt.index[0]]
                            selected_language = all_languages[evt.index[0]]
                            
                            # 更新选择历史
                            self.selection_history.add(selected_path)
                            self.current_selection = selected_path
                            
                            print(f"选择了预设音频: {selected_path}")
                            print(f"自动选择语言: {selected_language}")
                            print("音频已加载，请点击'开始识别'按钮进行识别")
                            
                            # 更新表格高亮显示
                            updated_table = update_table_highlighting(selected_path)
                            
                            # 返回选中的音频路径、音频播放器、自动选择的语言和更新的表格
                            return selected_path, selected_path, selected_language, updated_table
                        else:
                            print(f"选择的行索引超出范围: {evt.index[0]}")
                            return None, None, None, None
                    else:
                        print(f"点击的不是选择按钮列: {evt.index[1]}")
                        return None, None, None, None
                except Exception as e:
                    print(f"处理表格选择事件失败: {str(e)}")
                    return None, None, None, None
            
            def update_demo_audio_player(selected_audio):
                """更新预设音频播放器"""
                if selected_audio is None or selected_audio == "":
                    return None
                return selected_audio
            
            def update_table_highlighting(selected_audio):
                """更新表格高亮显示 - 使用背景色显示选择状态"""
                try:
                    # 重新构建表格数据，添加高亮显示
                    table_data = []
                    for lang in ["chinese", "english"]:
                        for audio_info in self.demo_audio_files[lang]:
                            # 获取音频信息
                            audio_info_dict = self.vad_processor.get_audio_info(audio_info["path"])
                            
                            # 检查音频状态 - 使用类的状态而不是参数
                            is_current_selected = audio_info["path"] == self.current_selection
                            is_historical_selected = audio_info["path"] in self.selection_history and not is_current_selected
                            
                            # 构建选择按钮文本（使用纯文本和符号）
                            if is_current_selected:
                                select_button = "🔴 已选择"  # 当前选择 - 红色圆点
                            elif is_historical_selected:
                                select_button = "🟡 已选择"  # 历史选择 - 黄色圆点
                            else:
                                select_button = "🎵 选择"    # 未选择 - 音符
                            
                            table_data.append([
                                audio_info["name"],
                                audio_info_dict["info_str"],
                                f"{'中文' if lang == 'chinese' else '英文'} | {select_button}"
                            ])
                    
                    return table_data
                except Exception as e:
                    print(f"更新表格高亮显示失败: {str(e)}")
                    return self._get_initial_demo_table_data()
            
            
            def process_demo_audio(selected_audio, language, use_vad):
                """处理预设音频"""
                if selected_audio is None:
                    return "请先选择预设音频", "等待选择"
                
                status_text = f"正在处理预设音频... (语言: {language})"
                result = self.process_audio(selected_audio, language, use_vad)
                
                if "识别失败" in result:
                    return result, "处理失败"
                else:
                    return result, "处理完成"
            
            def vad_trim_demo_audio(selected_audio):
                """VAD修剪预设音频"""
                if selected_audio is None:
                    return "请先选择预设音频", "等待选择", None
                
                try:
                    trimmed_path = self.vad_processor.trim_audio_with_vad(selected_audio)
                    if trimmed_path != selected_audio:
                        return f"VAD修剪完成！\n\n原文件: {selected_audio}\n修剪后: {trimmed_path}", "VAD修剪完成", trimmed_path
                    else:
                        return "未检测到语音活动或修剪失败", "VAD处理失败", selected_audio
                except Exception as e:
                    return f"VAD修剪失败: {str(e)}", "处理失败", selected_audio
            
            # 绑定事件
            recognize_btn.click(
                fn=process_recording,
                inputs=[audio, language, use_vad],
                outputs=[output, status]
            )
            
            vad_trim_btn.click(
                fn=vad_trim_audio,
                inputs=[audio],
                outputs=[output, status, audio]
            )
            
            restore_btn.click(
                fn=restore_original_audio,
                inputs=[audio],
                outputs=[output, status, audio]
            )
            
            # 上传音频事件绑定
            preprocess_btn.click(
                fn=preprocess_uploaded_audio,
                inputs=[audio_upload],
                outputs=[output, status, audio_upload]
            )
            
            recognize_upload_btn.click(
                fn=process_uploaded_audio,
                inputs=[audio_upload, language_upload, use_vad_upload, selected_demo_audio],
                outputs=[output, status]
            )
            
            vad_trim_upload_btn.click(
                fn=vad_trim_uploaded_audio,
                inputs=[audio_upload, selected_demo_audio],
                outputs=[output, status, audio_upload]
            )
            
            restore_upload_btn.click(
                fn=restore_uploaded_audio,
                inputs=[audio_upload, selected_demo_audio],
                outputs=[output, status, audio_upload]
            )
            
            # 预设音频事件绑定（表格在启动时自动加载）
            
            demo_audio_table.select(
                fn=on_table_select,
                inputs=[],
                outputs=[selected_demo_audio, demo_audio_player, language_upload, demo_audio_table]
            )
            
            # 当选择音频时，更新表格高亮显示
            selected_demo_audio.change(
                fn=update_table_highlighting,
                inputs=[selected_demo_audio],
                outputs=[demo_audio_table]
            )
            
            # 添加使用说明
            gr.Markdown("""
            ## 📖 使用说明
            
            ### 🎙️ 录音识别
            1. **选择语言**: 选择中文或English
            2. **VAD选项**: 可选择启用VAD修剪，自动去除静音部分
            3. **开始录音**: 点击录音按钮，允许浏览器访问麦克风
            4. **说话**: 对着麦克风清晰说话
            5. **停止录音**: 录音会自动停止
            6. **音频处理**: 
               - 点击"VAD修剪"按钮去除静音部分
               - 点击"恢复原音频"按钮恢复原始录音
            7. **开始识别**: 点击"开始识别"按钮
            8. **查看结果**: 识别结果将显示在右侧
            
            ### 📁 上传音频
            1. **预设音频**: 
               - 启动时自动加载预设音频表格，显示可用的预设音频
               - 表格显示：文件名、音频信息（采样率、时长、声道）、语言和选择按钮
               - 点击表格中的任意一行选择预设音频，可播放预览
               - 选择音频后自动设置识别语言（中文/英文）
               - 表格高亮显示：🔴 当前选择，🟡 历史选择，🎵 未选择
               - 选择音频后需要点击"开始识别"按钮进行识别
            2. **选择语言**: 选择中文或English（选择预设音频时自动设置）
            3. **VAD选项**: 可选择启用VAD修剪，自动去除静音部分
            4. **上传音频**: 点击上传按钮，选择音频文件（可选）
            5. **预处理音频**: 点击"预处理音频"按钮，转换为单声道16kHz（仅上传音频，保存到临时目录）
            6. **音频处理**: 
               - 点击"VAD修剪"按钮去除静音部分
               - 点击"恢复原音频"按钮恢复原始音频（仅上传音频）
            7. **开始识别**: 点击"开始识别"按钮（预设音频或上传音频都需要此步骤）
            8. **查看结果**: 识别结果将显示在右侧
            
            ## 🎯 VAD功能说明
            
            - **VAD修剪**: 自动检测语音活动，去除录音前后的静音部分
            - **提高准确性**: 修剪后的音频通常能获得更好的识别效果
            - **可恢复**: 支持恢复原始音频，方便对比效果
            
            ## 🔧 音频预处理说明
            
            - **自动转换**: 上传的音频自动转换为单声道16kHz格式
            - **格式支持**: 支持WAV、MP3、M4A等常见音频格式
            - **质量保证**: 预处理确保音频质量，提高识别准确性
            
            ## 🎵 预设音频说明
            
            - **预设音频**: 提供中文和英文的预设音频文件用于测试
            - **快速体验**: 无需录音或上传，直接体验语音识别功能
            - **多语言支持**: 包含中文和英文两种语言的预设音频
            - **音频预览**: 选择音频后可以播放预览，确认内容
            - **表格显示**: 以表格形式显示所有可用的预设音频，方便选择
            
            ## 💡 录音技巧
            
            - 保持安静的环境
            - 距离麦克风适中（10-20cm）
            - 说话清晰，语速适中
            - 避免背景噪音
            - 使用VAD修剪可提高识别准确性
            """)
        
        return interface
    
    def launch(self, host='0.0.0.0', port=7860, share=False, debug=False, ssl_certfile=None, ssl_keyfile=None):
        """启动Gradio 5.x应用"""
        print(f"启动RWKV ASR 录音识别系统 (Gradio 5.x)...")
        print(f"访问地址: http://{host}:{port}")
        if ssl_certfile and ssl_keyfile:
            print(f"使用SSL证书: {ssl_certfile} 和 {ssl_keyfile}")
            self.interface.launch(
                server_name=host,
                server_port=port,
                share=share,
                debug=debug,
                ssl_certfile=ssl_certfile,
                ssl_keyfile=ssl_keyfile
            )
        else:
            self.interface.launch(
                server_name=host,
                server_port=port,
                share=share,
                debug=debug
            )



@click.command()
@click.option('--audio-lm-path', default="/home/yueyulin/models/rwkv7_0.1b_audio_lm_latents_1.5b_202k", 
              help='音频语言模型路径')
@click.option('--llm-path', default="/home/yueyulin/models/rwkv7-g1a-1.5b-20250922-ctx4096.pth", 
              help='大语言模型路径')
@click.option('--whisper-path', default="/home/yueyulin/models/whisper-large-v3/", 
              help='Whisper模型路径')
@click.option('--tokenizer-path', default="tokenizer/rwkv_vocab_v20230424.txt", 
              help='分词器路径')
@click.option('--device', default="cuda:0", 
              help='设备类型 (cuda:0/cpu)')
@click.option('--dtype', default="float16", 
              type=click.Choice(['float16', 'float32', 'bfloat16']),
              help='数据类型')
@click.option('--host', default="0.0.0.0", 
              help='服务器主机地址')
@click.option('--port', default=7860, 
              help='服务器端口')
@click.option('--share', is_flag=True, 
              help='创建公共链接')
@click.option('--debug', is_flag=True, 
              help='启用调试模式')
@click.option('--ssl_certfile', default=None, 
              help='SSL证书文件路径')
@click.option('--ssl_keyfile', default=None, 
              help='SSL密钥文件路径')
def main(audio_lm_path, llm_path, whisper_path, tokenizer_path, device, dtype, host, port, share, debug, ssl_certfile, ssl_keyfile):
    """
    RWKV ASR Gradio 5.x 录音应用主程序
    """
    # 检查模型路径
    if not os.path.exists(audio_lm_path):
        print(f"错误: 音频语言模型路径不存在: {audio_lm_path}")
        return
    
    if not os.path.exists(llm_path):
        print(f"错误: 大语言模型路径不存在: {llm_path}")
        return
    
    if not os.path.exists(whisper_path):
        print(f"错误: Whisper模型路径不存在: {whisper_path}")
        return
    
    if not os.path.exists(tokenizer_path):
        print(f"错误: 分词器路径不存在: {tokenizer_path}")
        return
    
    # 创建并启动Gradio应用
    app = Gradio5ASR(
        audio_lm_path=audio_lm_path,
        llm_path=llm_path,
        whisper_path=whisper_path,
        tokenizer_path=tokenizer_path,
        device=device,
        dtype=dtype
    )
    
    app.launch(host=host, port=port, share=share, debug=debug, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)

if __name__ == "__main__":
    main()
