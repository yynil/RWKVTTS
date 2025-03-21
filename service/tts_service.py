import os
import time
import threading
import queue
import torch
import numpy as np
import traceback
import io
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import Future
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

class TTS_Service:
    def __init__(self, model_path, device_list, threads_per_device=2):
        """
        初始化TTS服务，为每个设备创建多个线程，每个线程都有一个CosyVoice2实例
        
        参数:
            model_path: 模型路径
            device_list: 设备列表，如 ['cuda:0', 'cuda:1']
            threads_per_device: 每个设备上的线程数
        """
        self.model_path = model_path
        self.device_list = device_list if isinstance(device_list, list) else [device_list]
        self.threads_per_device = threads_per_device
        self.total_threads = len(self.device_list) * threads_per_device
        
        # 任务队列
        self.task_queue = queue.Queue()
        
        # 启动工作线程
        self.workers = []
        self.stop_event = threading.Event()
        
        print(f"初始化TTS服务，模型路径: {model_path}")
        print(f"设备列表: {device_list}, 每个设备线程数: {threads_per_device}")
        print(f"总线程数: {self.total_threads}")
        # 启动工作线程
        for i in range(self.total_threads):
            device_idx = i % len(self.device_list)
            device = self.device_list[device_idx]
            worker_thread = threading.Thread(
                target=self._worker_loop,
                args=(i, device),
                daemon=True
            )
            self.workers.append(worker_thread)
            worker_thread.start()
    
    def _worker_loop(self, worker_id, device):
        """工作线程的主循环，处理TTS任务"""
        try:
            # 初始化CosyVoice2引擎
            print(f"工作线程 {worker_id} 正在初始化 CosyVoice2 引擎，设备: {device}")
            engine = CosyVoice2(self.model_path, device=device, fp16=False, load_jit=False)
            if worker_id == 0:
                self.speaker_ids = engine.frontend.spk2info.keys()
            print(f"工作线程 {worker_id} 已初始化完成，设备: {device}")
            
            while not self.stop_event.is_set():
                try:
                    # 从队列获取任务，设置超时以便检查stop_event
                    task = self.task_queue.get(timeout=0.5)
                    if task is None:
                        # 收到停止信号
                        self.task_queue.task_done()
                        break
                        
                    future, text, prompt_text, prompt_audio, audio_format,ref_voice = task
                    
                    try:
                        start_time = time.time()
                        
                        if isinstance(prompt_audio, bytes) and len(prompt_audio) > 0:
                            # 将字节流转换为音频数据
                            import soundfile as sf
                            with io.BytesIO(prompt_audio) as buffer:
                                prompt_data, sample_rate = sf.read(buffer)
                                # 确保是单声道，并转换为float32
                                if len(prompt_data.shape) > 1:
                                    prompt_data = prompt_data[:, 0]
                                prompt_data = prompt_data.astype(np.float32)
                                
                                # 重采样到16kHz (如果需要)
                                if sample_rate != 16000:
                                    from librosa import resample
                                    prompt_data = resample(prompt_data, orig_sr=sample_rate, target_sr=16000)
                                
                                prompt_speech_16k = torch.tensor(prompt_data).unsqueeze(0)
                        else:
                            # 无提示音频的情况
                            prompt_speech_16k = None
                            if ref_voice is not None:
                                if ref_voice in self.speaker_ids:
                                    ref_voice = ref_voice
                                else:
                                    raise ValueError(f"未找到说话人: {ref_voice}")
                        
                        # 运行TTS推理
                        tts_result = None
                        if prompt_speech_16k is None:
                            if prompt_text is not None and len(prompt_text) >0:
                                tts_text = f'{prompt_text}<|endofprompt|>{text}'
                            else:
                                tts_text = text
                            print(f'Processing {tts_text} from {ref_voice}')
                            for output in engine.inference_sft(tts_text, ref_voice, stream=False,speed=1):
                                tts_result = output['tts_speech']
                                break
                        else:
                            if prompt_text is not None and len(prompt_text.strip()) > 0:
                                for output in engine.inference_zero_shot(text, prompt_text, prompt_speech_16k, stream=False, speed=1):
                                    tts_result = output['tts_speech']
                                    break  # 只处理第一个输出
                            else:
                                for output in engine.inference_cross_lingual(text, prompt_speech_16k, stream=False, speed=1):
                                    tts_result = output['tts_speech']
                                    break  # 只处理第一个输出
                        
                        # 转换为字节流
                        audio_bytes = io.BytesIO()
                        if audio_format.lower() == "wav":
                            import soundfile as sf
                            sf.write(audio_bytes, tts_result.squeeze().cpu().numpy(), engine.sample_rate, format='WAV')
                        elif audio_format.lower() == "mp3":
                            import soundfile as sf
                            from pydub import AudioSegment
                            # 先保存为临时WAV
                            temp_wav = io.BytesIO()
                            sf.write(temp_wav, tts_result.squeeze().cpu().numpy(), engine.sample_rate, format='WAV')
                            temp_wav.seek(0)
                            # 转换为MP3
                            sound = AudioSegment.from_wav(temp_wav)
                            sound.export(audio_bytes, format="mp3")
                        
                        generation_time = time.time() - start_time
                        audio_time = tts_result.shape[1] / engine.sample_rate
                        
                        # 设置Future的结果
                        if not future.done():
                            future.set_result({
                                'audio': audio_bytes.getvalue(),
                                'generation_time': generation_time,
                                'audio_time': audio_time
                            })
                    
                    except Exception as e:
                        print(f"工作线程 {worker_id} 处理任务时出错: {str(e)}")
                        traceback.print_exc()
                        # 设置Future的异常
                        if not future.done():
                            future.set_exception(RuntimeError(f"TTS处理失败: {str(e)}"))
                    
                    # 标记任务完成
                    self.task_queue.task_done()
                    
                except queue.Empty:
                    # 队列为空，继续循环
                    continue
                except Exception as e:
                    print(f"工作线程 {worker_id} 处理任务循环中发生未预期错误: {str(e)}")
                    traceback.print_exc()
        
        except Exception as e:
            print(f"工作线程 {worker_id} 初始化或运行时出错: {str(e)}")
            traceback.print_exc()
    
    def tts(self, text: str, prompt_text: Optional[str] = None, 
            prompt_audio: Optional[bytes] = None, audio_format: str = "wav",
            timeout: float = 600.0,ref_voice: str=None) -> Dict[str, Any]:
        """
        执行文本到语音的转换
        
        参数:
            text: 需要转换的文本
            prompt_text: 提示文本 (可选)
            prompt_audio: 提示音频的字节流 (可选)
            audio_format: 输出音频格式 ('wav' 或 'mp3')
            timeout: 等待结果的超时时间（秒）
            
        返回:
            字典，包含生成的音频字节流、生成时间和音频时长
        """
        # 创建一个Future对象
        future = Future()
        
        # 将任务放入队列
        self.task_queue.put((future, text, prompt_text, prompt_audio, audio_format,ref_voice))
        
        try:
            # 等待Future完成或超时
            return future.result(timeout=timeout)
        except Exception as e:
            # 处理各种异常情况
            if isinstance(e, TimeoutError):
                raise TimeoutError(f"TTS处理超时 (> {timeout}秒)")
            else:
                # 重新抛出原始异常
                raise
    
    def shutdown(self):
        """关闭服务，停止所有工作线程"""
        print("正在关闭TTS服务...")
        self.stop_event.set()
        
        # 向队列发送停止信号
        for _ in range(self.total_threads):
            self.task_queue.put(None)
        
        # 等待所有线程完成
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        print("TTS服务已关闭")