#!/usr/bin/env python3
"""
改进的RefAudioUtilities类，使用新的ONNX模型，与BiCodecTokenizer完全一致
"""

import onnxruntime as ort
import numpy as np
import soundfile as sf
import soxr
from pathlib import Path
from typing import Tuple, Union, Optional


class RefAudioUtilitiesImproved:
    """改进的音频处理工具类，与BiCodecTokenizer完全一致"""
    
    def __init__(self, bicodec_onnx_path: str, wav2vec2_onnx_path: str,
                 ref_segment_duration: float = 6.0, latent_hop_length: int = 320):
        """
        初始化改进的ONNX模型
        
        Args:
            bicodec_onnx_path: BiCodec ONNX模型文件路径
            wav2vec2_onnx_path: Wav2Vec2 ONNX模型文件路径
            ref_segment_duration: 参考音频时长（秒）
            latent_hop_length: 潜在特征跳长度
        """
        # 加载BiCodec ONNX模型
        self.bicodec_session = ort.InferenceSession(bicodec_onnx_path)
        print(f"✅ BiCodec ONNX模型加载成功: {bicodec_onnx_path}")
        
        # 加载Wav2Vec2 ONNX模型
        self.wav2vec2_session = ort.InferenceSession(wav2vec2_onnx_path)
        print(f"✅ Wav2Vec2 ONNX模型加载成功: {wav2vec2_onnx_path}")
        
        # 基本配置
        self.sample_rate = 16000
        self.ref_segment_duration = ref_segment_duration
        self.latent_hop_length = latent_hop_length
        
        # 获取模型输入输出信息
        self.bicodec_input_names = [input_info.name for input_info in self.bicodec_session.get_inputs()]
        self.bicodec_output_names = [output_info.name for output_info in self.bicodec_session.get_outputs()]
        self.wav2vec2_input_names = [input_info.name for input_info in self.wav2vec2_session.get_inputs()]
        self.wav2vec2_output_names = [output_info.name for output_info in self.wav2vec2_session.get_outputs()]
        
        print(f"BiCodec模型输入: {self.bicodec_input_names}")
        print(f"BiCodec模型输出: {self.bicodec_output_names}")
        print(f"Wav2Vec2模型输入: {self.wav2vec2_input_names}")
        print(f"Wav2Vec2模型输出: {self.wav2vec2_output_names}")
    
    def load_audio(self, audio_path: Union[str, Path], target_sr: int = 16000, 
                   volume_normalize: bool = False) -> np.ndarray:
        """
        加载音频文件，与BiCodecTokenizer完全一致
        
        Args:
            audio_path: 音频文件路径
            target_sr: 目标采样率
            volume_normalize: 是否进行音量归一化
            
        Returns:
            音频数据数组
        """
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)
        
        # 使用soundfile加载音频，与BiCodecTokenizer完全一致
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]  # 取第一个通道
        
        # 重采样到目标采样率
        if sr != target_sr:
            audio = soxr.resample(audio, sr, target_sr, quality="VHQ")
            sr = target_sr
        
        # 音量归一化
        if volume_normalize:
            audio = self._audio_volume_normalize(audio)
        
        return audio
    
    def _audio_volume_normalize(self, audio: np.ndarray) -> np.ndarray:
        """音频音量归一化"""
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max()
        return audio
    
    def extract_wav2vec2_features(self, wav: np.ndarray) -> np.ndarray:
        """
        使用改进的ONNX Wav2Vec2模型提取特征，与BiCodecTokenizer完全一致
        
        Args:
            wav: 音频数据
            
        Returns:
            特征向量
        """
        try:
            # 准备输入：确保音频是float32类型，范围在[-1, 1]之间
            if wav.dtype != np.float32:
                wav = wav.astype(np.float32)
            
            # 如果音频值不在[-1, 1]范围内，进行归一化
            if np.abs(wav).max() > 1.0:
                wav = wav / np.abs(wav).max()
            
            # 添加batch维度
            input_data = wav[np.newaxis, :]  # [1, sequence_length]
            
            # 运行改进的Wav2Vec2推理
            inputs = {'wavs': input_data}
            outputs = self.wav2vec2_session.run(self.wav2vec2_output_names, inputs)
            
            # 输出形状应该是 [1, time_steps, 1024]
            features = outputs[0][0]  # 移除batch维度，得到 [time_steps, 1024]
            
            return features.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Wav2Vec2推理失败: {e}")
            raise
    
    def get_ref_clip(self, wav: np.ndarray) -> np.ndarray:
        """
        获取参考音频片段，与BiCodecTokenizer完全一致
        
        Args:
            wav: 原始音频数据
            
        Returns:
            参考音频片段
        """
        # 使用与BiCodecTokenizer相同的计算方式
        ref_segment_length = (
            int(self.sample_rate * self.ref_segment_duration)
            // self.latent_hop_length
            * self.latent_hop_length
        )
        wav_length = len(wav)
        
        if ref_segment_length > wav_length:
            # 如果音频不足指定长度，重复音频直到达到要求
            repeat_times = ref_segment_length // wav_length + 1
            wav = np.tile(wav, repeat_times)
        
        # 截取指定长度
        return wav[:ref_segment_length]
    
    def process_audio(self, audio_path: Union[str, Path], volume_normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理音频文件，返回原始音频和参考音频，与BiCodecTokenizer完全一致
        
        Args:
            audio_path: 音频文件路径
            volume_normalize: 是否进行音量归一化
            
        Returns:
            (原始音频, 参考音频)
        """
        wav = self.load_audio(audio_path, volume_normalize=volume_normalize)
        ref_wav = self.get_ref_clip(wav)
        
        return wav, ref_wav
    
    def tokenize(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用改进的ONNX模型生成tokens，与BiCodecTokenizer完全一致
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            (global_tokens, semantic_tokens)
        """
        # 处理音频
        wav, ref_wav = self.process_audio(audio_path, volume_normalize=False)
        
        # 提取Wav2Vec2特征
        feat = self.extract_wav2vec2_features(wav)
        
        # 准备BiCodec ONNX模型输入
        # 注意：新的BiCodec模型接受原始音频输入，不需要梅尔频谱图
        wav_input = wav[np.newaxis, :].astype(np.float32)  # [1, wav_length]
        ref_wav_input = ref_wav[np.newaxis, :].astype(np.float32)  # [1, ref_length]
        feat_input = feat[np.newaxis, :, :].astype(np.float32)  # [1, feat_length, 1024]
        
        # 运行BiCodec ONNX模型
        inputs = {
            'wav': wav_input,
            'ref_wav': ref_wav_input,
            'feat': feat_input
        }
        
        outputs = self.bicodec_session.run(self.bicodec_output_names, inputs)
        
        # 解析输出
        global_tokens = outputs[0]  # 第一个输出
        semantic_tokens = outputs[1]  # 第二个输出
        
        return global_tokens, semantic_tokens
    
    def tokenize_batch(self, audio_paths: list) -> Tuple[list, list]:
        """
        批量处理音频文件
        
        Args:
            audio_paths: 音频文件路径列表
            
        Returns:
            (global_tokens_list, semantic_tokens_list)
        """
        global_tokens_list = []
        semantic_tokens_list = []
        
        for audio_path in audio_paths:
            global_tokens, semantic_tokens = self.tokenize(audio_path)
            global_tokens_list.append(global_tokens)
            semantic_tokens_list.append(semantic_tokens)
        
        return global_tokens_list, semantic_tokens_list


# 测试函数
def test_improved_ref_audio_utilities():
    """测试改进的RefAudioUtilities类"""
    print("=== 测试改进的RefAudioUtilities类 ===")
    
    # 模型路径
    bicodec_onnx_path = '/Volumes/bigdata/models/BiCodec_original.onnx'
    wav2vec2_onnx_path = "/Volumes/bigdata/models/wav2vec2_improved.onnx"
    
    # 检查模型文件是否存在
    if not Path(bicodec_onnx_path).exists():
        print(f"❌ BiCodec ONNX模型不存在: {bicodec_onnx_path}")
        print("请先运行 convert_bicodec_to_onnx.py 生成模型")
        return
    
    if not Path(wav2vec2_onnx_path).exists():
        print(f"❌ Wav2Vec2 ONNX模型不存在: {wav2vec2_onnx_path}")
        print("请先运行 convert_wav2vec2_improved.py 生成模型")
        return
    
    try:
        # 初始化改进的工具类
        utilities = RefAudioUtilitiesImproved(
            bicodec_onnx_path=bicodec_onnx_path,
            wav2vec2_onnx_path=wav2vec2_onnx_path,
            ref_segment_duration=6.0,
            latent_hop_length=320
        )
        
        # 测试音频文件
        test_audio_path = "../demos/刘德华/dehua_zh.wav"
        
        if Path(test_audio_path).exists():
            print(f"✅ 测试音频文件: {test_audio_path}")
            
            # 生成tokens
            global_tokens, semantic_tokens = utilities.tokenize(test_audio_path)
            
            print(f"✅ Tokens生成成功!")
            print(f"   Global tokens shape: {global_tokens.shape}")
            print(f"   Semantic tokens shape: {semantic_tokens.shape}")
            print(f"   Global tokens: {global_tokens.flatten()[:10]}")
            print(f"   Semantic tokens (前10个): {semantic_tokens.flatten()[:10]}")
            
        else:
            print(f"❌ 测试音频文件不存在: {test_audio_path}")
            
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_improved_ref_audio_utilities()
