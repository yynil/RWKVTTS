import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Dict, Unpack, List, Callable, AnyStr
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils.deprecation import deprecate_kwarg
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Model, RWKV7ForCausalLM, Cache
from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
from cosyvoice.utils.common import IGNORE_ID
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
import time
from rwkvfla.models.rwkv7.configuration_rwkv7 import RWKV7Config

class RWKV7CosyConfig(RWKV7Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm_input_size = kwargs.get("llm_input_size", self.hidden_size)
        self.llm_output_size = kwargs.get("llm_output_size", self.hidden_size)
        self.speech_token_size = kwargs.get("speech_token_size", 6561)
        self.length_normalized_loss = kwargs.get("length_normalized_loss", True)
        self.lsm_weight = kwargs.get("lsm_weight", 0.0)
        self.mix_ratio = kwargs.get("mix_ratio", [5, 15])
        self.drop_ratio = kwargs.get("drop_ratio", 0.0)

class RWKV7CosyLM(RWKV7ForCausalLM):
    config_class = RWKV7CosyConfig
    
    def __init__(self, config: RWKV7CosyConfig):
        super().__init__(config)
        
        # 基础组件
        self.model = RWKV7Model(config)
        
        # 特殊标记
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2
        
        # 嵌入层
        self.llm_embedding = nn.Embedding(2, config.llm_input_size)
        self.text_embedding = nn.Embedding(config.vocab_size, config.llm_input_size)
        self.speech_embedding = nn.Embedding(config.speech_token_size + 1, config.llm_input_size)
        
        # 输出层 - 直接设置为语音标记大小
        self.lm_head = nn.Linear(config.hidden_size, config.speech_token_size + 1)
        
        # 损失函数
        self.criterion_ce = LabelSmoothingLoss(
            size=config.speech_token_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=config.lsm_weight,
            normalize_length=config.length_normalized_loss,
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.drop_ratio) if config.drop_ratio > 0 else None
        
        # 采样方法
        self.sampling = None
        self.mix_ratio = config.mix_ratio
        
        self.post_init()
        self.speech_token_size = config.speech_token_size

    def pad_unpad_sequence(self, sos_eos_emb, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        device = text_token.device
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_eos_emb.squeeze(dim=0), text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        attention_mask = [torch.ones(i.size(0), device=device, dtype=torch.int32) for i in lm_input]
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        return lm_input, attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # 处理批处理数据
        if 'batch' in kwargs:
            batch = kwargs['batch']
            text_token = batch['text_token']
            text_token_len = batch['text_token_len']
            speech_token = batch['speech_token']
            speech_token_len = batch['speech_token_len']

            # 准备目标
            lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token[i, :speech_token_len[i]].tolist() +
                                    [self.speech_token_size]) for i in range(text_token.size(0))]
            lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(text_token.device)

            # 编码文本
            text_token = self.text_embedding(text_token)

            # eos 和 task_id
            sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
            task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

            # 编码语音标记
            speech_token = self.speech_embedding(speech_token)

            # unpad 和 pad 序列
            inputs_embeds, attention_mask = self.pad_unpad_sequence(
                sos_eos_emb, text_token, text_token_len,
                task_id_emb, speech_token, speech_token_len
            )

            if self.dropout is not None:
                inputs_embeds = self.dropout(inputs_embeds)

            labels = lm_target[:, 1:].contiguous()

        # 模型前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.criterion_ce(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def sampling_ids(
        self,
        weighted_scores: torch.Tensor,
        decoded_tokens: List,
        sampling: int,
        ignore_eos: bool = True,
    ):
        num_trials, max_trials = 0, 100
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
            num_trials += 1
            if num_trials > max_trials:
                print(f'decoded_tokens is {decoded_tokens}, top_ids is {top_ids}, sampling is {sampling}, ignore_eos is {ignore_eos}')
                raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!'.format(max_trials))
        return top_ids

    @torch.inference_mode()
    def inference(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 0.5,
        cache = None
    ):
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len = text_len + prompt_text_len
        original_text_len = text_len.item()
        
        # 处理提示文本
        end_of_prompt_id = 65531
        end_of_prompt_mask = (text == end_of_prompt_id)
        end_of_prompt_indices = end_of_prompt_mask.nonzero()
        
        instruction_length = 0
        content_length = text_len
        
        if end_of_prompt_indices.size(0) > 0:
            instruction_length = end_of_prompt_indices[0, 1].item()
            content_length = text_len - (instruction_length + 1)
            original_text_len -= (instruction_length + 1)
    
        text_len += prompt_text_len
        text = self.text_embedding(text)
        
        # 准备输入
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.config.llm_input_size, dtype=text.dtype).to(device)
        
        lm_input = torch.concat([sos_eos_emb, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 计算长度限制
        min_len = content_length * min_token_text_ratio
        max_len = content_length * max_token_text_ratio
        
        # 逐步解码
        out_tokens = []
        start_time = time.time()
        is_prefill = True
        prefill_time = 0
        prefill_length = lm_input.shape[1]
        
        for i in range(max_len):
            logits, cache = self.forward_one_step(
                lm_input,
                masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                cache=cache
            )
            
            logp = logits[:,-1].log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i+original_text_len < min_len else False).item()
            
            if top_ids == self.speech_token_size:
                if cache and hasattr(cache, 'states'):
                    for layer_idx in range(len(cache.states)):
                        cache.states[layer_idx]['conv_state'] = torch.zeros_like(cache.states[layer_idx]['conv_state'])
                        cache.states[layer_idx]['ffn_state'] = torch.zeros_like(cache.states[layer_idx]['ffn_state'])
                print(f'finish decoding:{getattr(cache, "seen_tokens", 0)}')
                break
                
            if top_ids > self.speech_token_size:
                continue
                
            yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
            
            if is_prefill:
                prefill_time = time.time() - start_time
                is_prefill = False
                
        end_time = time.time()
        decode_time = end_time - start_time - prefill_time
        decoded_length = len(out_tokens)
        print(f'tps for prefill is {prefill_length} tokens in {prefill_time} seconds')
        print(f'tps for decode is {decoded_length} tokens in {decode_time} seconds')
        print(f'out_tokens is {out_tokens}')

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        logits = self.lm_head(outs.last_hidden_state)
        new_cache = outs.past_key_values
        return logits, new_cache

    def dummy_forward(self):
        print(f'start to do dummy forward')
        with torch.no_grad():
            with torch.amp.autocast(enabled=True, device_type='cuda'):
                xs = torch.ones(1, 1, self.config.llm_input_size, device=self.model.device, dtype=torch.float)
                print(f'xs is {xs.dtype}')
                masks = torch.ones(1, 1, 1, device=self.model.device, dtype=torch.long)
                cache = None
                self.forward_one_step(xs, masks, cache)
        print(f'finish dummy forward') 