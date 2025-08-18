import onnxruntime as ort
import numpy as np
import librosa
import soundfile as sf
import soxr
from pathlib import Path
from typing import Tuple, Union, Optional
import soundfile as sf


class RefAudioUtilities:
    """音频处理工具类，使用ONNX模型生成tokens"""
    
    def __init__(self, onnx_model_path: str, wav2vec2_path: str = None, 
                 ref_segment_duration: float = 6.0, latent_hop_length: int = 320):
        """
        初始化ONNX模型
        
        Args:
            onnx_model_path: ONNX模型文件路径
            wav2vec2_path: wav2vec2 ONNX模型文件路径，如果为None则不加载wav2vec2模型
            ref_segment_duration: 参考音频时长（秒）
            latent_hop_length: 潜在特征跳长度
        """
        self.ort_session = ort.InferenceSession(onnx_model_path)
        self.sample_rate = 16000
        self.ref_segment_duration = ref_segment_duration
        self.latent_hop_length = latent_hop_length
        
        # 获取模型输入输出信息
        self.input_names = [input_info.name for input_info in self.ort_session.get_inputs()]
        self.output_names = [output_info.name for output_info in self.ort_session.get_outputs()]
        
        print(f"模型输入: {self.input_names}")
        print(f"模型输出: {self.output_names}")
        
        # 初始化wav2vec2模型
        self.wav2vec2_session = None
        if wav2vec2_path:
            try:
                self.wav2vec2_session = ort.InferenceSession(wav2vec2_path)
                print("wav2vec2 ONNX模型加载成功")
            except Exception as e:
                print(f"wav2vec2 ONNX模型加载失败: {e}")
                self.wav2vec2_session = None
    
    def load_audio(self, audio_path: Union[str, Path], target_sr: int = 16000, 
                   volume_normalize: bool = False) -> np.ndarray:
        """
        加载音频文件，与BiCodecTokenizer保持一致
        
        Args:
            audio_path: 音频文件路径
            target_sr: 目标采样率
            volume_normalize: 是否进行音量归一化
            
        Returns:
            音频数据数组
        """
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)
        
        # 使用soundfile加载音频，与BiCodecTokenizer保持一致
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]  # 如果是立体声，取第一个通道
        
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
    
    def extract_mel_spectrogram(self, wav: np.ndarray, n_mels: int = 128, 
                               n_fft: int = 1024, hop_length: int = 256, 
                               win_length: int = 1024) -> np.ndarray:
        """
        提取梅尔频谱图
        
        Args:
            wav: 音频数据
            n_mels: 梅尔滤波器组数量
            n_fft: FFT窗口大小
            hop_length: 帧移
            win_length: 窗口长度
            
        Returns:
            梅尔频谱图
        """
        mel_spec = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=1,
            norm="slaney"
        )
        
        # 转换为dB单位
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 归一化到[0, 1]范围
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        return mel_spec_norm
    
    def extract_wav2vec2_features(self, wav: np.ndarray) -> np.ndarray:
        """
        使用ONNX wav2vec2模型提取特征，模拟BiCodecTokenizer的行为
        
        Args:
            wav: 音频数据
            
        Returns:
            特征向量
        """
        # 检查wav2vec2模型是否已加载
        if self.wav2vec2_session is None:
            raise RuntimeError("wav2vec2模型未加载，请在初始化时提供wav2vec2_path参数")
        
        try:
            # 准备输入：确保音频是float32类型，范围在[-1, 1]之间
            if wav.dtype != np.float32:
                wav = wav.astype(np.float32)
            
            # 如果音频值不在[-1, 1]范围内，进行归一化
            if np.abs(wav).max() > 1.0:
                wav = wav / np.abs(wav).max()
            
            # 添加batch维度
            input_data = wav[np.newaxis, :]  # [1, sequence_length]
            
            # 运行wav2vec2推理
            # 注意：这个ONNX模型已经包含了特征提取器的预处理和多个隐藏层的组合
            inputs = {'input_values': input_data}
            outputs = self.wav2vec2_session.run(None, inputs)
            
            # 输出形状应该是 [1, time_steps, 1024]
            # 这个输出已经是通过选择隐藏层11, 14, 16并计算平均值得到的
            features = outputs[0][0]  # 移除batch维度，得到 [time_steps, 1024]
            
            return features.astype(np.float32)
            
        except Exception as e:
            print(f"wav2vec2推理失败: {e}")
            print("回退到MFCC特征提取")
            return self._extract_mfcc_features(wav)
    
    def _extract_mfcc_features(self, wav: np.ndarray) -> np.ndarray:
        """
        回退方法：使用MFCC特征作为替代
        
        Args:
            wav: 音频数据
            
        Returns:
            特征向量
        """
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(
            y=wav, 
            sr=self.sample_rate, 
            n_mfcc=13,
            hop_length=320,  # 对应wav2vec2的步长
            n_fft=1024
        )
        
        # 提取delta和delta-delta特征
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # 组合特征
        features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
        
        # 转置并填充到1024维（模拟wav2vec2特征）
        features = features.T  # (time, features)
        
        # 如果特征维度不足1024，用零填充
        if features.shape[1] < 1024:
            padding = np.zeros((features.shape[0], 1024 - features.shape[1]))
            features = np.concatenate([features, padding], axis=1)
        else:
            features = features[:, :1024]
        
        return features.astype(np.float32)
    
    def get_ref_clip(self, wav: np.ndarray) -> np.ndarray:
        """
        获取参考音频片段，与BiCodecTokenizer保持一致
        
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
        处理音频文件，返回原始音频和参考音频，与BiCodecTokenizer保持一致
        
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
        使用ONNX模型生成tokens
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            (global_tokens, semantic_tokens)
        """
        # 处理音频
        wav, ref_wav = self.process_audio(audio_path)
        
        # 提取特征
        feat = self.extract_wav2vec2_features(wav)
        ref_mel = self.extract_mel_spectrogram(ref_wav)
        
        # 准备ONNX模型输入
        # 注意：根据ONNX模型信息，输入需要特定的形状
        # ref_wav_mel: [1, 128, 301]
        # feat: [1, feat_len, 1024]
        
        # 调整ref_mel形状
        if ref_mel.shape[1] < 301:
            # 如果时间维度不足301，用最后一帧重复填充
            padding = np.tile(ref_mel[:, -1:], (1, 301 - ref_mel.shape[1]))
            ref_mel = np.concatenate([ref_mel, padding], axis=1)
        else:
            ref_mel = ref_mel[:, :301]
        
        # 确保mel特征是128维
        if ref_mel.shape[0] != 128:
            if ref_mel.shape[0] < 128:
                # 如果mel维度不足128，用零填充
                padding = np.zeros((128 - ref_mel.shape[0], ref_mel.shape[1]))
                ref_mel = np.concatenate([ref_mel, padding], axis=0)
            else:
                ref_mel = ref_mel[:128, :]
        
        # 调整feat形状
        if feat.shape[1] < 1024:
            # 如果特征维度不足1024，用零填充
            padding = np.zeros((feat.shape[0], 1024 - feat.shape[1]))
            feat = np.concatenate([feat, padding], axis=1)
        else:
            feat = feat[:, :1024]
        
        # 添加batch维度
        ref_mel_input = ref_mel[np.newaxis, :, :].astype(np.float32)  # [1, 128, 301]
        feat_input = feat[np.newaxis, :, :].astype(np.float32)  # [1, feat_len, 1024]
        
        # 运行ONNX模型
        inputs = {
            'ref_wav_mel': ref_mel_input,
            'feat': feat_input
        }
        
        outputs = self.ort_session.run(self.output_names, inputs)
        
        # 解析输出
        semantic_tokens = outputs[0]  # 第一个输出
        global_tokens = outputs[1]    # 第二个输出
        
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
def test_ref_audio_utilities():
    """测试RefAudioUtilities类"""
    # 初始化工具类
    onnx_model_path = '/Volumes/bigdata/models/BiCodecTokenize.onnx'
    wav2vec2_path = "/Volumes/bigdata/models/wav2vec2-large-xlsr-53.onnx"
    # 使用与BiCodecTokenizer相同的参数
    utilities = RefAudioUtilities(
        onnx_model_path, 
        wav2vec2_path,
        ref_segment_duration=6.0,  # 6秒参考音频
        latent_hop_length=320       # 潜在特征跳长度
    )
    
    # 测试音频文件（使用项目中的示例音频）
    test_audio_path = "demos/刘德华/dehua_zh.wav"
    
    if Path(test_audio_path).exists():
        print(f"测试音频文件: {test_audio_path}")
        
        try:
            # 生成tokens
            global_tokens, semantic_tokens = utilities.tokenize(test_audio_path)
            
            print(f"Global tokens shape: {global_tokens.shape}")
            print(f"Semantic tokens shape: {semantic_tokens.shape}")
            print(f"Global tokens: {global_tokens}")
            print(f"Semantic tokens (前10个): {semantic_tokens.flatten()[:10]}")
            
        except Exception as e:
            print(f"处理音频时出错: {e}")
    else:
        print(f"测试音频文件不存在: {test_audio_path}")
        print("请确保测试音频文件存在")


if __name__ == "__main__":
    test_ref_audio_utilities()
