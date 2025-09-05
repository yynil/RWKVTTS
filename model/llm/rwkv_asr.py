import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Config
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Model
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM
from transformers import AutoTokenizer
import os
class RWKV7ASRModel(nn.Module):
    def __init__(self, audio_lm_model:RWKV7Model, llm:RWKV7ForCausalLM):
        super().__init__()
        self.audio_lm_model = audio_lm_model
        self.projector = nn.Linear(audio_lm_model.config.hidden_size, llm.config.hidden_size)
        self.llm = llm

    def forward(self, audio_input_ids, text_input_ids, audio_attention_mask, text_attention_mask, labels=None, labels_attention_mask=None, hints_ids=None):
        """
        重新设计的forward方法，按照正确的逻辑处理数据
        
        Args:
            audio_input_ids: 左对齐的音频tokens [B, T_audio]
            audio_attention_mask: 音频attention mask [B, T_audio]
            text_input_ids: 左对齐的指令文本tokens [B, T_text]
            text_attention_mask: 文本attention mask [B, T_text]
            labels: 目标标签 [B, T_labels]
            labels_attention_mask: 标签attention mask [B, T_labels]
            hints_ids: 提示词tokens [T_hints] 或 [B, T_hints]
        """
        batch_size = audio_input_ids.shape[0]
        
        # 1. 生成音频嵌入
        audio_latents = self.audio_lm_model(audio_input_ids, audio_attention_mask, use_cache=False, return_dict=False)[0]  # [B, T_audio, hidden_size]
        projected_latents = self.projector(audio_latents)  # [B, T_audio, hidden_size_of_llm]
        
        # 2. 生成文本嵌入
        text_input_embeds = self.llm.get_input_embeddings()(text_input_ids)  # [B, T_text, hidden_size_of_llm]
        
        # 3. 处理hints_ids：如果是一维的，扩展成 (B, T_hints)
        if hints_ids is not None:
            if hints_ids.dim() == 1:
                hints_ids = hints_ids.unsqueeze(0).expand(batch_size, -1)
            hints_embeds = self.llm.get_input_embeddings()(hints_ids)  # [B, T_hints, hidden_size_of_llm]
        else:
            hints_embeds = None
        
        # 4. 生成标签嵌入（如果提供）
        if labels is not None and labels_attention_mask is not None:
            cloned_labels = labels.clone()
            #set -100 to 0
            cloned_labels[cloned_labels == -100] = 0
            labels_embeds = self.llm.get_input_embeddings()(cloned_labels)  # [B, T_labels, hidden_size_of_llm]
        else:
            labels_embeds = None
        
        # 5. 遍历所有样本，根据attention mask连接有效部分
        valid_embeds_list = []
        valid_attention_mask_list = []
        valid_labels_list = []
        
        # 添加调试信息和错误处理
        if not hasattr(self, 'debug_printed'):
            print(f"Debug: audio_attention_mask shape: {audio_attention_mask.shape}")
            print(f"Debug: audio_attention_mask dtype: {audio_attention_mask.dtype}")
            print(f"Debug: audio_attention_mask device: {audio_attention_mask.device}")
            # 移除有问题的调试信息
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
            audio_valid_lengths = audio_attention_mask.sum(dim=1)
            text_valid_lengths = text_attention_mask.sum(dim=1)
            labels_valid_lengths = labels_attention_mask.sum(dim=1)
        except Exception as e:
            print(f"Error in attention mask sum: {e}")
            print(f"audio_attention_mask shape: {audio_attention_mask.shape}")
            print(f"text_attention_mask shape: {text_attention_mask.shape}")
            print(f"labels_attention_mask shape: {labels_attention_mask.shape}")
            raise e
        
        for i in range(batch_size):
            # 获取当前样本的有效长度
            audio_valid_length = audio_valid_lengths[i].item()
            text_valid_length = text_valid_lengths[i].item()
            labels_valid_length = labels_valid_lengths[i].item()
            

            
            # 获取有效的音频嵌入（左padding，有效元素在右边）
            audio_valid_embeds = projected_latents[i, -audio_valid_length:] if audio_valid_length > 0 else torch.empty(0, projected_latents.size(-1), device=projected_latents.device, dtype=projected_latents.dtype)
            
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
                if labels_valid_length > labels_embeds.size(1):
                    print(f"Warning: labels_valid_length {labels_valid_length} > labels_embeds.size(1) {labels_embeds.size(1)}")
                    labels_valid_length = labels_embeds.size(1)
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
        
        # 6. 使用pad_sequence进行左对齐
        input_embeds = pad_sequence(valid_embeds_list, batch_first=True, padding_value=0.0,padding_side='left')
        attention_mask = pad_sequence(valid_attention_mask_list, batch_first=True, padding_value=0,padding_side='left')
        final_labels = pad_sequence(valid_labels_list, batch_first=True, padding_value=-100,padding_side='left')
        
        # 调试信息（只在第一个batch打印）
        if not hasattr(self, "first_batch"):
            print(f'input_embeds shape: {input_embeds.shape}')
            print(f'attention_mask shape: {attention_mask.shape}')
            print(f'labels shape: {final_labels.shape}')
            print(f'labels sample: {final_labels}')  # 显示第一个样本的前50个标签
            self.first_batch = True
        
        # 7. 调用LLM
        if labels is not None:
            output = self.llm(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=final_labels)
        else:
            output = self.llm(inputs_embeds=input_embeds, attention_mask=attention_mask)
        
        return output

    @torch.inference_mode()
    def inference_single(self, audio_tokens, text_tokens, hints_tokens):
        audio_attention_mask = torch.ones((audio_tokens.shape[0], audio_tokens.shape[1]),dtype=torch.long).to(audio_tokens.device)
        audio_latents = self.audio_lm_model(audio_tokens, audio_attention_mask, use_cache=False, return_dict=False)[0]  # [B, T_audio, hidden_size]
        projected_latents = self.projector(audio_latents)  # [B, T_audio, hidden_size_of_llm]
        text_input_embeds = self.llm.get_input_embeddings()(text_tokens)  # [B, T_text, hidden_size_of_llm]
        hints_embeds = self.llm.get_input_embeddings()(hints_tokens)  # [B, T_hints, hidden_size_of_llm]
        combined_embeds = torch.cat([text_input_embeds, projected_latents, hints_embeds], dim=1)  # [B, T_total, hidden_size_of_llm]
        attention_mask = torch.ones((combined_embeds.shape[0], combined_embeds.shape[1]),dtype=torch.long).to(combined_embeds.device)
        gen_args = {
            "inputs_embeds": combined_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": 512,
            "temperature": 1.0,
            "top_k": 10,
            "top_p": 0.8,
            "do_sample": True,
            "eos_token_id": 0,
            "max_length": 2048
        }
        output = self.llm.generate(**gen_args)
        return output[0][:-1].tolist()

if __name__ == "__main__":
    llm_path = "/home/yueyulin/models/rwkv7-0.4B-g1"
    audio_model_path = "/home/yueyulin/models/rwkv7_0.1b_audio_lm/"
    llm = RWKV7ForCausalLM.from_pretrained(llm_path )
    audio_model = RWKV7Model.from_pretrained(audio_model_path)
    asr_model = RWKV7ASRModel(audio_model, llm)
    device = "cuda:0"
    asr_model = asr_model.to(torch.bfloat16).to(device)
    print(asr_model)

    audio_input_ids = torch.randint(0, 8192, (2, 10)).to(device)
    audio_attention_mask = torch.ones((2, 10),dtype=torch.long).to(device)
    audio_attention_mask[0,0:5] = 0
    text_input_ids = torch.randint(0, 65536, (2, 10)).to(device)
    text_attention_mask = torch.ones((2, 10),dtype=torch.long).to(device)
    text_attention_mask[0,1:2] = 0
    labels = torch.randint(0, 65536, (2, 10)).to(device)
    labels_attention_mask = torch.ones((2, 10),dtype=torch.long).to(device)
    labels_attention_mask[1,0:3] = 0
    hints_ids = torch.randint(0, 65536, (10,)).to(device)
    output = asr_model(audio_input_ids, text_input_ids, audio_attention_mask, text_attention_mask, labels, labels_attention_mask, hints_ids)
    print(output)