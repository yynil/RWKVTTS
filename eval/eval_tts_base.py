from abc import ABC, abstractmethod
import os
import torchaudio
import soundfile as sf
from librosa import resample
import numpy as np
import os
os.environ["RWKV_V7_ON"] = "1" # enable this for rwkv-7 models
os.environ['RWKV_JIT_ON'] = '1'  # 禁用 JIT 编译
os.environ["RWKV_CUDA_ON"] = '1'
from rwkv.model import RWKV
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from transformers import AutoTokenizer
from torch.nn import functional as F
from math import inf
import torch
class BaseTTSEngine(ABC):
    """
    TTS 引擎基础类，定义 TTS 操作的接口
    可以被不同的 TTS 实现类继承
    """
    
    def __init__(self, device: str = "cuda:0", **kwargs):
        """
        初始化 TTS 引擎
        
        Args:
            model_path: 模型路径
            device: 设备类型 (cuda:0, cpu 等)
        """
        self.device = device
        self.sample_rate = 16000  # 默认采样率
        self.kwargs = kwargs
        self._init_engine()
    
    @abstractmethod
    def _init_engine(self):
        """
        初始化具体的 TTS 引擎
        子类必须实现此方法
        """
        pass
    
    @abstractmethod
    def do_tts(self, 
                tts_text: str, 
                prompt_text: str, 
                prompt_audio_file: str, 
                final_output_file: str) -> bool:
        """
        执行 TTS 操作
        
        Args:
            tts_text: 要转换为语音的文本
            prompt_text: 提示文本
            prompt_audio_file: 提示音频文件路径
            final_output_file: 最终输出音频文件路径
            
        Returns:
            bool: 操作是否成功
        """
        pass
    
    def save_audio(self, audio_tensor, output_file: str) -> bool:
        """
        保存音频文件
        
        Args:
            audio_tensor: 音频张量
            output_file: 输出文件路径
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            torchaudio.save(output_file, audio_tensor, self.sample_rate)
            return True
        except Exception as e:
            print(f"保存音频文件失败: {e}")
            return False
    
    def load_prompt_audio(self, audio_file: str, target_sr: int = 16000):
        """
        加载提示音频文件
        
        Args:
            audio_file: 音频文件路径
            target_sr: 目标采样率
            
        Returns:
            加载的音频数据
        """
        prompt_audio, sampling_rate = sf.read(audio_file)
        if sampling_rate != target_sr:
            prompt_audio = resample(prompt_audio, orig_sr=sampling_rate, target_sr=target_sr)
            prompt_audio = np.array(prompt_audio, dtype=np.float32)
        return prompt_audio

def sample_logits(logits, temperature=1.0, top_p=0.85, top_k=0,black_list_tokens=[]):
    if temperature == 0:
        temperature = 1.0
        top_p = 0
    probs = F.softmax(logits.float(), dim=-1)
    top_k = int(top_k)
    if black_list_tokens is not None:
        probs[black_list_tokens] = 0
    # 'privateuseone' is the type of custom devices like `torch_directml.device()`
    if probs.device.type in ['cpu', 'privateuseone']:
        probs = probs.cpu().numpy()
        sorted_ids = np.argsort(probs)
        sorted_probs = probs[sorted_ids][::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
        probs[probs < cutoff] = 0
        if top_k < len(probs) and top_k > 0:
            probs[sorted_ids[:-top_k]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        probs = probs / np.sum(probs)
        out = np.random.choice(a=len(probs), p=probs)
        return int(out)
    else:
        sorted_ids = torch.argsort(probs)
        sorted_probs = probs[sorted_ids]
        sorted_probs = torch.flip(sorted_probs, dims=(0,))
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
        probs[probs < cutoff] = 0
        if top_k < len(probs) and top_k > 0:
            probs[sorted_ids[:-top_k]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        out = torch.multinomial(probs, num_samples=1)[0]
        return int(out)
    
class ResparkTTSEngine(BaseTTSEngine):
    def _init_engine(self):
        model_path = self.kwargs.get('model_path')
        audio_tokenizer_path = self.kwargs.get('audio_tokenizer_path')
        self.language = self.kwargs.get('language')
        
        strategy = f'{self.device} bf16'
        
        self.model = RWKV(model=f"{model_path}/model_converted", strategy=strategy)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.audio_tokenizer = BiCodecTokenizer(model_dir=audio_tokenizer_path, device=self.device)
        self.target_sample_rate = self.audio_tokenizer.config['sample_rate']

    def do_tts(self, tts_text: str, prompt_text: str, prompt_audio_file: str, final_output_file: str,min_length_in_seconds: float=0.5) -> bool:
        try:
            print(f'tts_text: {tts_text}')
            print(f'prompt_text: {prompt_text}')
            print(f'prompt_audio_file: {prompt_audio_file}')
            print(f'final_output_file: {final_output_file}')
            print('-'*100)
            prompt_audio = self.load_prompt_audio(prompt_audio_file, self.target_sample_rate)
            prompt_global_tokens, prompt_semantic_tokens = self.audio_tokenizer.tokenize(prompt_audio)
            prompt_global_tokens = prompt_global_tokens.squeeze(0).squeeze(0).detach().cpu().tolist()
            prompt_semantic_tokens = prompt_semantic_tokens.squeeze(0).detach().cpu().tolist()
            TTS_TAG_0 = 8193
            TTS_TAG_1 = 8194
            TTS_TAG_2 = 8195
            prompt_global_tokens = [i + 8196 for i in prompt_global_tokens]
            text_tokens = self.tokenizer.encode(prompt_text+tts_text, add_special_tokens=False)
            text_tokens = [i + 8196+4096 for i in text_tokens]
            all_idx = [TTS_TAG_2] + text_tokens + [TTS_TAG_0] + prompt_global_tokens + [TTS_TAG_1] + prompt_semantic_tokens
   
            
            x,state = self.model.forward(all_idx, None)
            output_semantic_tokens = []
            min_tokens_num = int(min_length_in_seconds * 50)#50hz
            for i in range(2048):
                black_list = None if i > min_tokens_num else [8192]
                sampled_id = sample_logits(x, temperature=1.0, top_p=0.95, top_k=80,black_list_tokens=black_list)
                if sampled_id == 8192:
                    break
                output_semantic_tokens.append(sampled_id)
                x,state = self.model.forward([sampled_id], state)
                
            
            output_dir = os.path.dirname(final_output_file)
            os.makedirs(output_dir, exist_ok=True)
            global_tokens = torch.tensor([[i - 8196 for i in prompt_global_tokens]], dtype=torch.int32, device=self.device)
            semantic_tokens = torch.tensor([output_semantic_tokens], dtype=torch.int32, device=self.device)
            wav_reconstructed = self.audio_tokenizer.detokenize(global_tokens, semantic_tokens)
            sf.write(final_output_file, wav_reconstructed, self.target_sample_rate)
            
            # 清理中间变量
            del x, state, global_tokens, semantic_tokens, wav_reconstructed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            raise e
    
# 工厂函数，用于创建 TTS 引擎实例
def create_tts_engine(engine_type: str = "respark", **kwargs) -> BaseTTSEngine:
    """
    创建 TTS 引擎实例
    
    Args:
        engine_type: 引擎类型 ("respark", "cosyvoice" 等)
        **kwargs: 传递给引擎构造函数的参数
        
    Returns:
        TTS 引擎实例
    """
    if engine_type.lower() == "respark":
        return ResparkTTSEngine(**kwargs)
    else:
        raise ValueError(f"不支持的 TTS 引擎类型: {engine_type}")


# 示例用法
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='/home/yueyulin/models/rwkv7-0.4B-g1-respark-voice-tunable-25k/')
    parser.add_argument("--audio_tokenizer_path", type=str, default='/home/yueyulin/models/Spark-TTS-0.5B/')
    parser.add_argument("--device", type=str, default='cuda:0')
    args = parser.parse_args()
    print(args)

    # 创建 Respark 引擎实例
    engine = create_tts_engine(
        engine_type="respark",
        device=args.device,
        model_path=args.model_path,
        audio_tokenizer_path=args.audio_tokenizer_path,
        language='zh'
    )

    tts_text = '北京在出行规模，城市影响力方面表现优异。'
    tts_text = '你好'
    prompt_text = '空气又冷又潮湿，道路经常结冰。'
    prompt_audio_file = 'seedtts_testset/zh/prompt-wavs/10004152-00000092.wav'
    final_output_file = 'eval_results/zh/10004152-00000044.wav'
    engine.do_tts(tts_text=tts_text, 
                  prompt_text=prompt_text, 
                  prompt_audio_file=prompt_audio_file, 
                  final_output_file=final_output_file,
                  min_length_in_seconds=0.1)