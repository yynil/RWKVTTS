import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Config
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Model
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM

class RWKV7ASRModel(nn.Module):
    def __init__(self, audio_lm_model:RWKV7Model, llm:RWKV7ForCausalLM):
        super().__init__()
        self.audio_lm_model = audio_lm_model
        self.projector = nn.Linear(audio_lm_model.config.hidden_size, llm.config.hidden_size)
        self.llm = llm

    def forward(self, audio_input_ids, text_input_ids, audio_attention_mask, text_attention_mask, labels=None, hints_ids=None):
        audio_latents = self.audio_lm_model(audio_input_ids, audio_attention_mask,use_cache=False,return_dict=False)[0] #B,audio_seq_len,hidden_size
        projected_latents = self.projector(audio_latents) #B,audio_seq_len,hidden_size_of_llm
        text_input_embeds = self.llm.get_input_embeddings()(text_input_ids) #B,text_seq_len,hidden_size_of_llm
        
        batch_size = projected_latents.shape[0]
        
        # 计算每个样本的实际长度
        audio_lengths = audio_attention_mask.sum(dim=1)  # [B]
        text_lengths = text_attention_mask.sum(dim=1)    # [B]
        
        # 存储每个样本的有效嵌入和 labels
        valid_embeds_list = []
        valid_labels_list = []
        valid_attention_mask_list = []
        
        # 处理 hints_ids：如果是一维的，扩展成 (B, T_of_hints)
        if hints_ids is not None and hints_ids.dim() == 1:
            hints_ids = hints_ids.unsqueeze(0).expand(batch_size, -1)
        
        for i in range(batch_size):
            # 获取当前样本的有效音频嵌入（左 padding，有效元素在右边）
            audio_valid_length = audio_lengths[i]
            audio_valid_embeds = projected_latents[i, -audio_valid_length:] if audio_valid_length > 0 else torch.empty(0, projected_latents.size(-1), device=projected_latents.device, dtype=projected_latents.dtype)
            
            # 获取当前样本的有效文本嵌入（左 padding，有效元素在右边）
            text_valid_length = text_lengths[i]
            text_valid_embeds = text_input_embeds[i, -text_valid_length:] if text_valid_length > 0 else torch.empty(0, text_input_embeds.size(-1), device=text_input_embeds.device, dtype=text_input_embeds.dtype)
            
            # 获取 hints 嵌入（如果提供）
            hints_embeds = None
            if hints_ids is not None:
                hints_embeds = self.llm.get_input_embeddings()(hints_ids[i]) if hints_ids[i].numel() > 0 else torch.empty(0, text_input_embeds.size(-1), device=text_input_embeds.device, dtype=text_input_embeds.dtype)
            
            # 按照新顺序连接：text_input_embs, audio_input_embeds, hints_embeds
            embed_parts = [text_valid_embeds, audio_valid_embeds]
            if hints_embeds is not None:
                embed_parts.append(hints_embeds)
            combined_embeds = torch.cat(embed_parts, dim=0)
            valid_embeds_list.append(combined_embeds)
            valid_attention_mask_list.append(torch.ones(len(combined_embeds),dtype=torch.long,device=text_attention_mask.device))
            len_of_expansion = len(combined_embeds) - labels.shape[1]
            if len_of_expansion > 0:
                expanded = torch.tensor([-100]*len_of_expansion,device=labels.device,dtype=labels.dtype)
                final_labels = torch.cat([expanded,labels[i]],dim=0)
                valid_labels_list.append(final_labels)
            else:
                valid_labels_list.append(labels[i][-len_of_expansion:])
        
        # 使用 pad_sequence 进行左 padding
        input_embeds = pad_sequence(valid_embeds_list, batch_first=True, padding_value=0.0,padding_side="left")
        attention_mask = pad_sequence(valid_attention_mask_list, batch_first=True, padding_value=0.0,padding_side="left")
        final_labels = pad_sequence(valid_labels_list, batch_first=True, padding_value=-100,padding_side="left")
        if not hasattr(self,"first_batch"):
            print(f'input_embeds: {input_embeds}')
            print(f'attention_mask: {attention_mask}')
            print(f'labels: {final_labels}')
            print(f"input_embeds: {input_embeds.shape}")
            print(f"attention_mask: {attention_mask.shape}")
            print(f"labels: {final_labels.shape}")
            self.first_batch = True
        # 调用 LLM
        if labels is not None:
            output = self.llm(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=final_labels)
        else:
            output = self.llm(inputs_embeds=input_embeds, attention_mask=attention_mask)
        
        return output
    
    

if __name__ == "__main__":
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
        "vocab_size": 8192
    }
    config = RWKV7Config(**audio_config)
    model = RWKV7Model(config)
    print(model)
    model = model.to(torch.bfloat16).to("cuda")

    audio_input_ids = torch.randint(0, 8192, (1, 10)).to("cuda")
    audio_attention_mask = torch.ones((1, 10),dtype=torch.long).to("cuda")
    audio_output = model.forward(input_ids=audio_input_ids, attention_mask=audio_attention_mask,use_cache=False,return_dict=False)
    print(audio_output)
    print(audio_output[0].shape)
    llm_path = "/home/yueyulin/models/rwkv7-0.4B-g1"
    llm = RWKV7ForCausalLM.from_pretrained(llm_path)
    llm = llm.to(torch.bfloat16).to("cuda")
    text_input_ids = torch.randint(0, 65536, (1, 10)).to("cuda")
    text_attention_mask = torch.ones((1, 10),dtype=torch.long).to("cuda")
    text_output = llm.forward(input_ids=text_input_ids, attention_mask=text_attention_mask,use_cache=False,return_dict=False)
    print(text_output)
    print(text_output[0].shape)
    asr_model = RWKV7ASRModel(model,llm)
    asr_model = asr_model.to(torch.bfloat16).to("cuda")
    labels = torch.randint(0, 65536, (1, 10)).to("cuda")
    output = asr_model.forward(audio_input_ids, text_input_ids, audio_attention_mask, text_attention_mask,labels)
    print(output)
    print(output[0].shape)
