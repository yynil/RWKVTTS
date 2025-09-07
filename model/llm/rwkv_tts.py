import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Config
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Model
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM

class RWKV7TTSModel(nn.Module):
    def __init__(self, text_lm_model:RWKV7Model, audio_lm:RWKV7ForCausalLM):
        super().__init__()
        self.text_lm_model = text_lm_model  # 只需要生成hidden states
        self.audio_lm = audio_lm  # 完整的生成模型，包含lm_head
        
        # 音频模型的投影层，将文本模型的hidden_size映射到音频模型的hidden_size
        self.text_to_audio_projector = nn.Linear(text_lm_model.config.hidden_size, audio_lm.config.hidden_size)

    def gradient_checkpointing_enable(self,enable=True):
        self.audio_lm.gradient_checkpointing_enable()

    def forward(self, text_input_ids, text_attention_mask, audio_token_ids, audio_token_attention_mask, labels):
        """
        训练模式的forward函数
        Args:
            text_input_ids: 文本输入token ids [B, T_of_text]
            text_attention_mask: 文本注意力掩码 [B, T_of_text]
            audio_token_ids: 音频token ids [B, T_of_audio]，包含全局token和语义token
            audio_token_attention_mask: 音频token注意力掩码 [B, T_of_audio]
            labels: 标签 [B, T_of_audio]，用于计算损失
        """
        batch_size = text_input_ids.shape[0]
        
        # 1. 通过文本模型获取文本嵌入
        text_outputs = self.text_lm_model(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            use_cache=False,
            return_dict=False
        )
        text_hidden_states = text_outputs[0]  # [B, text_seq_len, text_hidden_size]
        
        # 2. 投影文本嵌入到音频模型的维度
        projected_text_embeds = self.text_to_audio_projector(text_hidden_states)  # [B, text_seq_len, audio_hidden_size]
        
        # 3. 计算文本序列的实际长度
        text_lengths = text_attention_mask.sum(dim=1)  # [B]
        
        # 4. 准备音频模型的输入
        valid_embeds_list = []
        valid_attention_mask_list = []
        valid_labels_list = []
        
        for i in range(batch_size):
            # 获取当前样本的有效文本嵌入（左padding，有效元素在右边）
            text_valid_length = text_lengths[i]
            text_valid_embeds = projected_text_embeds[i, -text_valid_length:] if text_valid_length > 0 else torch.empty(0, projected_text_embeds.size(-1), device=projected_text_embeds.device, dtype=projected_text_embeds.dtype)
            
            # 获取音频token嵌入（只取有效的token）
            audio_valid_length = audio_token_attention_mask[i].sum()
            audio_valid_tokens = audio_token_ids[i, -audio_valid_length:] if audio_valid_length > 0 else torch.empty(0, dtype=audio_token_ids.dtype, device=audio_token_ids.device)
            audio_token_embeds = self.audio_lm.get_input_embeddings()(audio_valid_tokens) if audio_valid_length > 0 else torch.empty(0, projected_text_embeds.size(-1), device=projected_text_embeds.device, dtype=projected_text_embeds.dtype)
            
            # 合并所有嵌入：文本 + 音频token
            combined_embeds = torch.cat([text_valid_embeds, audio_token_embeds], dim=0)
            valid_embeds_list.append(combined_embeds)
            
            # 创建注意力掩码：所有有效嵌入对应的位置都是1
            combined_mask = torch.ones(combined_embeds.size(0), dtype=torch.long, device=text_attention_mask.device)
            valid_attention_mask_list.append(combined_mask)
            
            # 准备标签：文本部分为-100（不计算损失），只预测有效的音频token
            text_labels = torch.full((text_valid_length,), -100, device=text_input_ids.device, dtype=torch.long)
            # 只取有效的音频token标签（左padding，有效标签在右边）
            valid_audio_labels = labels[i][-audio_valid_length:] if audio_valid_length > 0 else torch.empty(0, dtype=labels.dtype, device=labels.device)
            combined_labels = torch.cat([text_labels, valid_audio_labels], dim=0)
            valid_labels_list.append(combined_labels)
        
        # 使用pad_sequence进行左padding
        input_embeds = pad_sequence(valid_embeds_list, batch_first=True, padding_value=0.0, padding_side="left")
        attention_mask = pad_sequence(valid_attention_mask_list, batch_first=True, padding_value=0, padding_side="left")
        final_labels = pad_sequence(valid_labels_list, batch_first=True, padding_value=-100, padding_side="left")
        
        # 调试信息
        if not hasattr(self, "first_batch"):
            print(f'input_embeds: {input_embeds.shape}')
            print(f'attention_mask: {attention_mask.shape}')
            print(f'labels: {final_labels.shape}')
            print(f'audio_token_ids: {audio_token_ids.shape}')
            print(f'audio_token_attention_mask: {audio_token_attention_mask.shape}')
            self.first_batch = True
        
        # 5. 通过音频模型生成
        audio_outputs = self.audio_lm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=final_labels,
            use_cache=False,
            return_dict=True
        )
        
        # 音频模型已经包含了lm_head，直接返回结果
        return audio_outputs

    def generate(self, text_input_ids, text_attention_mask, max_semantic_tokens=1024, temperature=1.0):
        """
        生成语音token
        Args:
            text_input_ids: 文本输入
            text_attention_mask: 文本注意力掩码
            max_semantic_tokens: 最大语义token数量
            temperature: 生成温度
        """
        self.eval()
        with torch.no_grad():
            batch_size = text_input_ids.shape[0]
            
            # 1. 获取文本嵌入
            text_outputs = self.text_lm_model(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
                use_cache=False,
                return_dict=False
            )
            text_hidden_states = text_outputs[0]
            projected_text_embeds = self.text_to_audio_projector(text_hidden_states)
            
            # 2. 准备初始输入
            text_lengths = text_attention_mask.sum(dim=1)
            
            # 3. 生成语义token
            semantic_tokens = []
            for i in range(batch_size):
                text_valid_length = text_lengths[i]
                text_valid_embeds = projected_text_embeds[i, -text_valid_length:]
                
                # 构建输入序列（只包含文本嵌入）
                input_sequence = text_valid_embeds
                
                # 生成语义token
                current_semantic_tokens = []
                for _ in range(max_semantic_tokens):
                    # 通过音频模型
                    audio_output = self.audio_lm(
                        inputs_embeds=input_sequence.unsqueeze(0),
                        use_cache=False,
                        return_dict=False
                    )
                    
                    # 获取logits
                    logits = audio_output[1] if len(audio_output) > 1 else None
                    if logits is None:
                        # 如果没有logits，使用音频模型的lm_head
                        hidden_states = audio_output[0]
                        logits = self.audio_lm.lm_head(hidden_states[:, -1:])  # 只取最后一个位置
                    
                    # 生成token
                    probs = torch.softmax(logits / temperature, dim=-1)
                    token = torch.multinomial(probs.view(-1), 1)
                    
                    if token.item() == 8192:  # EOS token
                        break
                    
                    current_semantic_tokens.append(token.item())
                    
                    # 更新输入序列（添加新生成的token嵌入）
                    token_embed = self.audio_lm.get_input_embeddings()(token)
                    input_sequence = torch.cat([input_sequence, token_embed], dim=0)
                
                semantic_tokens.append(current_semantic_tokens)
            
            return {
                'semantic_tokens': semantic_tokens
            }


if __name__ == "__main__":
    # 测试代码
    # 音频模型配置（包含embeddings和lm_head）
    audio_config = {
        "a_low_rank_dim": 64,
        "attn": None,
        "attn_mode": "chunk",
        "bos_token_id": 0,
        "decay_low_rank_dim": 64,
        "eos_token_id": 0,
        "fuse_cross_entropy": True,
        "fuse_norm": False,
        "gate_low_rank_dim": 128,
        "head_dim": 64,
        "hidden_act": "sqrelu",
        "hidden_ratio": 4.0,
        "hidden_size": 768,
        "initializer_range": 0.006,
        "intermediate_size": 3072,
        "max_position_embeddings": 2048,
        "model_type": "rwkv7",
        "norm_bias": True,
        "norm_eps": 1e-05,
        "norm_first": True,
        "num_heads": 32,
        "num_hidden_layers": 12,
        "tie_word_embeddings": False,
        "transformers_version": "4.48.0",
        "use_cache": True,
        "v_low_rank_dim": 32,
        "vocab_size": 12289  # 4096 (global) + 8193 (semantic)
    }
    
    # 初始化模型
    audio_config = RWKV7Config(**audio_config)
    audio_model = RWKV7ForCausalLM(audio_config)  # 使用完整的生成模型
    
    # 创建文本模型（只需要RWKV7Model）
    text_config = {
        "vocab_size": 65536,
        "hidden_size": 1024,
        "num_hidden_layers": 12,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "max_position_embeddings": 2048,
        "model_type": "rwkv7"
    }
    text_config = RWKV7Config(**text_config)
    text_model = RWKV7Model(text_config)  # 只需要模型，不需要lm_head
    
    # 创建TTS模型
    tts_model = RWKV7TTSModel(text_model, audio_model)
    tts_model = tts_model.to(torch.bfloat16).to("cuda")
    
    # 测试输入
    text_input_ids = torch.randint(0, 65536, (1, 20)).to("cuda")
    text_attention_mask = torch.ones((1, 20), dtype=torch.long).to("cuda")
    
    # 测试前向传播（训练模式）
    global_token_ids = torch.randint(0, 4096, (1, 32)).to("cuda")  # [B, Fixed_T=32]
    semantic_token_ids = torch.randint(4096, 12289, (1, 50)).to("cuda")  # [B, T_of_semantic]
    semantic_tokens_attention_mask = torch.ones((1, 50), dtype=torch.long).to("cuda")  # [B, T_of_semantic]
    labels = torch.randint(0, 12289, (1, 82)).to("cuda")  # [B, Fixed_T+T_of_semantic=32+50]
    
    outputs = tts_model.forward(text_input_ids, text_attention_mask, global_token_ids, semantic_token_ids, semantic_tokens_attention_mask, labels)
    print(f"Loss: {outputs}")
    
   