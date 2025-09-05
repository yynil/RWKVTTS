import io
from typing import OrderedDict
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Config
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Model
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM
import torchaudio
from transformers import AutoTokenizer,WhisperFeatureExtractor,WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder
import os


from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7PreTrainedModel, RWKV7Model
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Block
from rwkvfla.modules import  LayerNorm
class RWKV7ModelForLatentInputs(RWKV7Model):
    """
    一个专用于接收连续 latent 输入 (inputs_embeds) 的 RWKV7Model 版本。
    它不包含 nn.Embedding 层，以节省内存和显存。
    """
    def __init__(self, config: RWKV7Config):
        super(RWKV7Model, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([RWKV7Block(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (LayerNorm if config.fuse_norm else nn.LayerNorm)(
            config.hidden_size,
            bias=config.norm_bias,
            eps=config.norm_eps
        )

        self.gradient_checkpointing = False

        self.post_init()

    def forward(self, inputs_embeds, attention_mask=None, use_cache=False, return_dict=None):
        # 这个 forward 方法只接受 inputs_embeds，更清晰也更安全

        return super().forward(inputs_embeds=inputs_embeds,
                               attention_mask=attention_mask,
                               use_cache=use_cache,
                               return_dict=return_dict)

class RWKV7ASRModel(nn.Module):
    def __init__(self, whisper_encoder: WhisperEncoder, 
                audio_lm_model: RWKV7ModelForLatentInputs,
            llm: RWKV7ForCausalLM, whisper_feature_extractor: WhisperFeatureExtractor):
        super().__init__()
        self.whisper_encoder = whisper_encoder
        self.whisper_feature_extractor = whisper_feature_extractor
        self.projector1 = nn.Linear(whisper_encoder.config.hidden_size, audio_lm_model.config.hidden_size)
        self.audio_lm_model = audio_lm_model
        self.projector2 = nn.Linear(audio_lm_model.config.hidden_size, llm.config.hidden_size)
        self.llm = llm

    def forward(self, audio_data, text_input_ids, text_attention_mask, labels=None, labels_attention_mask=None, hints_ids=None):
        """
        重新设计的forward方法，按照正确的逻辑处理数据，参照rwkv_asr.py的格式
        
        Args:
            audio_data: 原始音频数据列表
            text_input_ids: 左对齐的指令文本tokens [B, T_text]
            text_attention_mask: 文本attention mask [B, T_text]
            labels: 目标标签 [B, T_labels]
            labels_attention_mask: 标签attention mask [B, T_labels]
            hints_ids: 提示词tokens [T_hints] 或 [B, T_hints]
        """
        batch_size = len(audio_data)
        
        # 1. 使用 whisper_feature_extractor 处理原始音频
        list_of_audio = []
        for audio in audio_data:
            if len(audio.shape) == 2:
                list_of_audio.append(audio.squeeze(0))
            else:
                list_of_audio.append(audio)
        
        features = self.whisper_feature_extractor(list_of_audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True, padding_value=0.0)
        audio_attention_mask = features['attention_mask'].squeeze(0)
        
        # 确保张量在正确的设备上
        device = next(self.whisper_encoder.parameters()).device
        input_features = features['input_features'].squeeze(0).to(dtype=torch.bfloat16).to(device)
        audio_attention_mask = audio_attention_mask.to(device)
        
        # 2. 通过 whisper_encoder 编码音频特征
        with torch.no_grad():
            encoder_outputs = self.whisper_encoder(input_features, attention_mask=audio_attention_mask)
        
        audio_latents = encoder_outputs.last_hidden_state  # [B, T_audio, hidden_size]
        projected_latents = self.projector1(audio_latents)  # [B, T_audio, hidden_size_of_llm]
        projected_latents = self.audio_lm_model(projected_latents, use_cache=False, return_dict=False)[0]  # [B, T_audio, hidden_size]
        projected_latents = self.projector2(projected_latents)  # [B, T_audio, hidden_size_of_llm]
        
        # 3. 生成文本嵌入
        text_input_embeds = self.llm.get_input_embeddings()(text_input_ids)  # [B, T_text, hidden_size_of_llm]
        
        # 4. 处理hints_ids：如果是一维的，扩展成 (B, T_hints)
        if hints_ids is not None:
            if hints_ids.dim() == 1:
                hints_ids = hints_ids.unsqueeze(0).expand(batch_size, -1)
            hints_embeds = self.llm.get_input_embeddings()(hints_ids)  # [B, T_hints, hidden_size_of_llm]
        else:
            hints_embeds = None
        
        # 5. 生成标签嵌入（如果提供）- 关键修复：处理-100值
        if labels is not None and labels_attention_mask is not None:
            cloned_labels = labels.clone()
            # 将-100设置为0，避免embedding溢出
            cloned_labels[cloned_labels == -100] = 0
            labels_embeds = self.llm.get_input_embeddings()(cloned_labels)  # [B, T_labels, hidden_size_of_llm]
        else:
            labels_embeds = None
        
        # 6. 计算下采样比例，处理attention mask不匹配问题
        # WhisperEncoder内部有下采样，将原始特征压缩到更少的帧
        if audio_attention_mask.shape[1] != audio_latents.shape[1]:
            # 计算下采样比例
            downsample_ratio = audio_attention_mask.shape[1] / audio_latents.shape[1]
            if not hasattr(self, 'downsample_printed'):
                print(f"Whisper下采样比例: {downsample_ratio:.2f} ({audio_attention_mask.shape[1]} -> {audio_latents.shape[1]})")
                self.downsample_printed = True
        else:
            downsample_ratio = 1.0
        
        # 7. 遍历所有样本，根据attention mask连接有效部分
        valid_embeds_list = []
        valid_attention_mask_list = []
        valid_labels_list = []
        
        # 添加调试信息和错误处理
        if not hasattr(self, 'debug_printed'):
            print(f"Debug: audio_attention_mask shape: {audio_attention_mask.shape}")
            print(f"Debug: audio_attention_mask dtype: {audio_attention_mask.dtype}")
            print(f"Debug: audio_attention_mask device: {audio_attention_mask.device}")
            print(f"Debug: audio_latents shape: {audio_latents.shape}")
            print(f"Debug: downsample_ratio: {downsample_ratio}")
            self.debug_printed = True
        
        # 确保数据类型正确
        if audio_attention_mask.dtype != torch.long:
            audio_attention_mask = audio_attention_mask.long()
        if text_attention_mask.dtype != torch.long:
            text_attention_mask = text_attention_mask.long()
        if labels_attention_mask.dtype != torch.long:
            labels_attention_mask = labels_attention_mask.long()
        
        # 添加安全的索引检查
        try:
            # 使用原始attention mask计算有效长度，然后除以下采样比例
            audio_valid_lengths = audio_attention_mask.sum(dim=1)
            text_valid_lengths = text_attention_mask.sum(dim=1)
            labels_valid_lengths = labels_attention_mask.sum(dim=1) if labels_attention_mask is not None else torch.zeros(batch_size, dtype=torch.long, device=device)
        except Exception as e:
            print(f"Error in attention mask sum: {e}")
            print(f"audio_attention_mask shape: {audio_attention_mask.shape}")
            print(f"text_attention_mask shape: {text_attention_mask.shape}")
            print(f"labels_attention_mask shape: {labels_attention_mask.shape if labels_attention_mask is not None else 'None'}")
            raise e
        
        for i in range(batch_size):
            # 获取当前样本的有效长度
            # 关键修复：将原始attention mask计算的长度除以下采样比例
            audio_valid_length = int(audio_valid_lengths[i].item() / downsample_ratio)
            text_valid_length = text_valid_lengths[i].item()
            labels_valid_length = labels_valid_lengths[i].item() if labels_attention_mask is not None else 0
            
            # 获取有效的音频嵌入（右padding，有效元素在左边）
            audio_valid_embeds = projected_latents[i, :audio_valid_length] if audio_valid_length > 0 else torch.empty(0, projected_latents.size(-1), device=projected_latents.device, dtype=projected_latents.dtype)
            
            # 获取有效的文本嵌入（左padding，有效元素在右边）
            text_valid_embeds = text_input_embeds[i, -text_valid_length:] if text_valid_length > 0 else torch.empty(0, text_input_embeds.size(-1), device=text_input_embeds.device, dtype=text_input_embeds.dtype)
            
            # 获取hints嵌入
            hints_valid_embeds = None
            if hints_embeds is not None:
                hints_valid_embeds = hints_embeds[i]  # [T_hints, hidden_size]
            
            # 获取标签嵌入
            labels_valid_embeds = None
            if labels_embeds is not None and labels_attention_mask is not None:
                labels_valid_length = labels_attention_mask[i].sum().item()
                labels_valid_embeds = labels_embeds[i, -labels_valid_length:] if labels_valid_length > 0 else torch.empty(0, labels_embeds.size(-1), device=labels_embeds.device, dtype=labels_embeds.dtype)
            
            # 按照顺序连接：text_embeds + audio_embeds + hints_embeds + labels_embeds
            embed_parts = [text_valid_embeds, audio_valid_embeds]
            if hints_valid_embeds is not None:
                embed_parts.append(hints_valid_embeds)
            if labels_valid_embeds is not None:
                embed_parts.append(labels_valid_embeds)
            
            combined_embeds = torch.cat(embed_parts, dim=0)  # [T_total, hidden_size]
            valid_embeds_list.append(combined_embeds)
            
            # 生成全1的attention mask
            valid_attention_mask = torch.ones(len(combined_embeds), dtype=torch.long, device=audio_attention_mask.device)
            valid_attention_mask_list.append(valid_attention_mask)
            
            # 生成labels：只对labels部分计算损失，其他部分设为-100
            if labels is not None and labels_attention_mask is not None:
                # 创建全-100的tensor
                sample_labels = torch.full((len(combined_embeds),), -100, dtype=labels.dtype, device=labels.device)
                
                # 只对labels部分赋值（由于左对齐padding，labels总是在最右边）
                if len(labels_valid_embeds) > 0:
                    labels_len = len(labels_valid_embeds)
                    sample_labels[-labels_len:] = labels[i, -labels_len:]
                
                valid_labels_list.append(sample_labels)
            else:
                # 如果没有labels，创建全-100的tensor
                sample_labels = torch.full((len(combined_embeds),), -100, dtype=torch.long, device=audio_attention_mask.device)
                valid_labels_list.append(sample_labels)
        
        # 7. 使用pad_sequence进行左对齐
        input_embeds = pad_sequence(valid_embeds_list, batch_first=True, padding_value=0.0, padding_side='left')
        attention_mask = pad_sequence(valid_attention_mask_list, batch_first=True, padding_value=0, padding_side='left')
        final_labels = pad_sequence(valid_labels_list, batch_first=True, padding_value=-100, padding_side='left')
        
        # 调试信息（只在第一个batch打印）
        if not hasattr(self, "first_batch"):
            print(f'input_embeds shape: {input_embeds.shape}')
            print(f'attention_mask shape: {attention_mask.shape}')
            print(f'labels shape: {final_labels.shape}')
            print(f'labels sample: {final_labels}')  # 显示第一个样本的前50个标签
            self.first_batch = True
        
        # 8. 调用LLM
        if labels is not None:
            output = self.llm(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=final_labels)
        else:
            output = self.llm(inputs_embeds=input_embeds, attention_mask=attention_mask)
        
        return output

    @torch.inference_mode()
    def inference_single(self, audio_data, text_tokens, hints_tokens):
        """
        单样本推理方法，使用原始音频数据而不是预处理的音频tokens
        
        Args:
            audio_data: 原始音频数据 [T_audio] 或 [1, T_audio]
            text_tokens: 文本tokens [1, T_text]
            hints_tokens: 提示词tokens [1, T_hints]
        
        Returns:
            generated_tokens: 生成的token列表
        """
        # 1. 处理音频数据格式
        if len(audio_data.shape) == 2:
            audio_data = audio_data.squeeze(0)
        
        # 2. 使用 whisper_feature_extractor 处理原始音频
        features = self.whisper_feature_extractor([audio_data], sampling_rate=16000, return_tensors="pt", return_attention_mask=True, padding_value=0.0)
        audio_attention_mask = features['attention_mask'].squeeze(0)
        
        # 确保张量在正确的设备上
        device = next(self.whisper_encoder.parameters()).device
        input_features = features['input_features'].squeeze(0).to(dtype=torch.bfloat16).to(device)
        audio_attention_mask = audio_attention_mask.to(device)
        
        # 3. 通过 whisper_encoder 编码音频特征
        with torch.no_grad():
            encoder_outputs = self.whisper_encoder(input_features, attention_mask=audio_attention_mask)
        
        audio_latents = encoder_outputs.last_hidden_state  # [1, T_audio, hidden_size]
        projected_latents = self.projector1(audio_latents)  # [1, T_audio, hidden_size_of_audio_lm]
        projected_latents = self.audio_lm_model(projected_latents, use_cache=False, return_dict=False)[0]  # [1, T_audio, hidden_size]
        projected_latents = self.projector2(projected_latents)  # [1, T_audio, hidden_size_of_llm]
        
        # 4. 计算下采样比例并获取有效音频数据
        # WhisperEncoder内部有下采样，将原始特征压缩到更少的帧
        if audio_attention_mask.shape[1] != audio_latents.shape[1]:
            # 计算下采样比例
            downsample_ratio = audio_attention_mask.shape[1] / audio_latents.shape[1]
        else:
            downsample_ratio = 1.0
        
        # 获取有效音频长度（考虑下采样）
        audio_valid_length = int(audio_attention_mask.sum().item() / downsample_ratio)
        
        # 获取有效的音频嵌入（右padding，有效元素在左边）
        if audio_valid_length > 0:
            audio_valid_embeds = projected_latents[0, :audio_valid_length]  # [T_valid, hidden_size]
        else:
            audio_valid_embeds = torch.empty(0, projected_latents.size(-1), device=projected_latents.device, dtype=projected_latents.dtype)
        
        # 5. 生成文本和hints嵌入
        text_input_embeds = self.llm.get_input_embeddings()(text_tokens)  # [1, T_text, hidden_size_of_llm]
        hints_embeds = self.llm.get_input_embeddings()(hints_tokens)  # [1, T_hints, hidden_size_of_llm]
        
        # 6. 连接所有embeddings: text + audio + hints
        # 注意：这里需要处理维度匹配，因为audio_valid_embeds是[T_valid, hidden_size]
        if len(audio_valid_embeds) > 0:
            # 将audio_valid_embeds扩展为batch维度
            audio_valid_embeds = audio_valid_embeds.unsqueeze(0)  # [1, T_valid, hidden_size]
            combined_embeds = torch.cat([text_input_embeds, audio_valid_embeds, hints_embeds], dim=1)  # [1, T_total, hidden_size_of_llm]
        else:
            # 如果没有有效音频数据，只连接文本和hints
            combined_embeds = torch.cat([text_input_embeds, hints_embeds], dim=1)  # [1, T_total, hidden_size_of_llm]
        attention_mask = torch.ones((combined_embeds.shape[0], combined_embeds.shape[1]), dtype=torch.long).to(combined_embeds.device)
        
        # 7. 生成参数
        gen_args = {
            "inputs_embeds": combined_embeds,
            "attention_mask": attention_mask,
            "max_length": 512,
            "temperature": 1.0,
            "top_k": 10,
            "top_p": 0.8,
            "do_sample": True,
            "eos_token_id": 0
        }
        
        # 8. 调用LLM生成
        output = self.llm.generate(**gen_args)
        return output[0][:-1].tolist()

def load_whisper_feature_extractor_and_encoder(whisper_path):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_path)
    print(f"Loaded WhisperFeatureExtractor: {feature_extractor}")
    whisper_config = WhisperConfig.from_pretrained(whisper_path)
    print(f"Loaded WhisperConfig: {whisper_config}")
    encoder = WhisperEncoder(whisper_config)
    print(f"Created WhisperEncoder: {encoder}")
    
    # 加载预训练权重
    full_model_state_dict = torch.load(os.path.join(whisper_path, "pytorch_model.bin"),map_location=torch.device("cpu"))
    encoder_state_dict = OrderedDict()
    encoder_prefix = "model.encoder."
    for key, value in full_model_state_dict.items():
        if key.startswith(encoder_prefix):
            new_key = key[len(encoder_prefix):]
            encoder_state_dict[new_key] = value
    encoder.load_state_dict(encoder_state_dict)
    print(f"Loaded encoder weights from {whisper_path}")
    
    return feature_extractor, encoder

