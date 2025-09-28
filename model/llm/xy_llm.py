import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List

from transformers.modeling_outputs import CausalLMOutputWithPast
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Model, RWKV7ForCausalLM, Cache
from rwkvfla.models.rwkv7.configuration_rwkv7 import RWKV7Config

# Generation imports
from transformers.generation import GenerationMixin, LogitsProcessorList, StoppingCriteriaList, GenerationConfig
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.generation.streamers import BaseStreamer


class RWKV7XYConfig(RWKV7Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm_input_size = kwargs.get("llm_input_size", self.hidden_size)
        self.speech_vocab_size = kwargs.get("speech_vocab_size", 1024)
        self.length_normalized_loss = kwargs.get("length_normalized_loss", True)
        self.lsm_weight = kwargs.get("lsm_weight", 0.0)
        self.num_channels = kwargs.get("num_channels", 8)
        self.drop_ratio = kwargs.get("drop_ratio", 0.0)
        self.speech_pad_token = kwargs.get("speech_pad_token", self.speech_vocab_size - 1)
        # text_shift_size is critical for generation logic
        self.text_shift_size = kwargs.get("text_shift_size", 65536)


class CustomGenerationMixin(GenerationMixin):
    """
    Custom GenerationMixin to provide a bespoke _sample method for the RWKV7XYLM model.
    """
    def is_audio_token(self, token_id: torch.Tensor) -> torch.Tensor:
        """Checks if a token is a valid audio token for Channel 0."""
        return (token_id >= self.config.text_shift_size) & (token_id < self.config.text_shift_size + self.config.speech_vocab_size)

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateDecoderOnlyOutput, torch.LongTensor]:
        
        # --- 1. Initialization ---
        speech_pad_idx = self.config.speech_pad_token
        text_shift_size = self.config.text_shift_size
        speech_vocab_size = self.config.speech_vocab_size
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        output_scores = generation_config.output_scores
        return_dict_in_generate = generation_config.return_dict_in_generate
        
        batch_size, cur_len, channels = input_ids.shape
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        # -1: normal generation, >0: flushing countdown
        needs_additional_steps = -1 * torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        
        # The `logits_processor` is the main one for text. We will handle audio sampling manually.
        text_logits_processor = logits_processor

        # --- 2. Decode Loop ---
        while True:
            # --- 2a. Prepare model inputs ---
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            
            # --- 2b. Forward pass ---
            outputs = self(**model_inputs, return_dict=True)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # --- 2c. Get logits for all 8 channels ---
            next_token_logits = [logits[:, -1, :].clone().float() for logits in outputs.logits]

            # --- 2d. **CRITICAL LOGIC**: Constrain Channel 0 logits ---
            # Force Channel 0 to only sample tokens in the audio range [65536, 65536+1024)
            # by setting all other token logits to -inf.
            mask = torch.ones_like(next_token_logits[0])
            mask[:, text_shift_size : text_shift_size + speech_vocab_size] = 0
            next_token_logits[0].masked_fill_(mask.bool(), -float("inf"))

            # Apply standard logits processors (like temperature) to all channels
            next_token_scores = [text_logits_processor(input_ids[..., i], logits) for i, logits in enumerate(next_token_logits)]
            
            # --- 2e. Sample next tokens for all channels ---
            next_tokens_list = []
            for i, channel_score in enumerate(next_token_scores):
                probs = F.softmax(channel_score, dim=-1)
                channel_ntk = torch.multinomial(probs, num_samples=1).squeeze(1)
                next_tokens_list.append(channel_ntk)
            next_tokens = torch.stack(next_tokens_list, dim=-1)

            # --- 2f. Flushing Logic ---
            # Check if Channel 0 produced a non-audio token, which triggers the flushing countdown.
            is_audio = self.is_audio_token(next_tokens[:, 0])
            indices_to_flush = (~is_audio) & (needs_additional_steps < 0)
            if indices_to_flush.any():
                needs_additional_steps[indices_to_flush] = channels - 1 # Start the 7-step countdown

            # During the countdown, force output to EOS for text and PAD for audio channels.
            is_flushing = needs_additional_steps >= 0
            if is_flushing.any():
                if eos_token_id is not None:
                    next_tokens[is_flushing, 0] = eos_token_id[0]
                for i in range(1, channels):
                    # Pad channel `i` if its final token has been generated
                    pad_this_channel = is_flushing & (needs_additional_steps < channels - i)
                    next_tokens[pad_this_channel, i] = speech_pad_idx
            
            # --- 2g. Update state for next iteration ---
            # Ensure finished sequences continue to output pad/eos tokens
            if eos_token_id is not None:
                pddp_text = eos_token_id[0]
            else:
                pddp_text = 0 # Default pad token
            next_tokens[:, 0] = next_tokens[:, 0] * unfinished_sequences + pddp_text * (1 - unfinished_sequences)
            next_tokens[:, 1:] = next_tokens[:, 1:] * unfinished_sequences.unsqueeze(-1) + speech_pad_idx * (1 - unfinished_sequences.unsqueeze(-1))

            input_ids = torch.cat([input_ids, next_tokens[:, None, :]], dim=1)
            if streamer is not None:
                streamer.put(next_tokens[:, 0].cpu())
            
            # Update countdown and check for stopping criteria
            needs_additional_steps[is_flushing] -= 1
            stopping_criteria_met = stopping_criteria(input_ids[..., 0], None)
            unfinished_sequences = unfinished_sequences & ~stopping_criteria_met & ~(needs_additional_steps == -1)
            
            if unfinished_sequences.max() == 0:
                break

        # --- 3. Finalize and Return ---
        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(sequences=input_ids)
        else:
            return input_ids


class RWKV7XYLM(RWKV7ForCausalLM, CustomGenerationMixin):
    config_class = RWKV7XYConfig
    
    def __init__(self, config: RWKV7XYConfig):
        super().__init__(config)
        
        self.model = RWKV7Model(config)
        
        self.embs = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.criterions = nn.ModuleList()
        
        # Channel 0: text
        self.embs.append(nn.Embedding(config.vocab_size, config.hidden_size,padding_idx=config.vocab_size-1))
        self.heads.append(nn.Linear(config.hidden_size, config.vocab_size))
        self.criterions.append(nn.CrossEntropyLoss(label_smoothing=config.lsm_weight))
        
        # Channels 1 to num_channels-1: speech
        for _ in range(1, config.num_channels):
            self.embs.append(nn.Embedding(config.speech_vocab_size, config.hidden_size,padding_idx=config.speech_vocab_size-1))
            self.heads.append(nn.Linear(config.hidden_size, config.speech_vocab_size))
            self.criterions.append(nn.CrossEntropyLoss(label_smoothing=config.lsm_weight))

        self.dropout = nn.Dropout(config.drop_ratio) if config.drop_ratio > 0 else None
        
        self.post_init()

    def zero_embs(self):
        """
        Manually zero out the embedding vectors for the padding indices.
        """
        text_pad_idx = self.config.vocab_size - 1
        if self.embs[0].padding_idx is not None:
            self.embs[0].weight.data[text_pad_idx].zero_()

        speech_pad_idx = self.config.speech_vocab_size - 1
        for i in range(1, self.config.num_channels):
            if self.embs[i].padding_idx is not None:
                self.embs[i].weight.data[speech_pad_idx].zero_()

    def forward(
        self,
        input_ids: torch.LongTensor = None, # (B, T, num_channels)
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        labels: Optional[torch.LongTensor] = None, # (B, T, num_channels)
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if inputs_embeds is None and input_ids is not None:
            if input_ids.dim() != 3 or input_ids.shape[2] != self.config.num_channels:
                raise ValueError(f"input_ids must have shape (B, T, num_channels), but got {input_ids.shape}")

            B, T, num_channels = input_ids.shape
            inputs_embeds = torch.zeros(B, T, self.config.hidden_size, device=input_ids.device, dtype=self.embs[0].weight.dtype)
            all_embeds = []
            for i in range(num_channels):
                channel_tokens = input_ids[:, :, i]
                channel_embeds = self.embs[i](channel_tokens)
                all_embeds.append(channel_embeds)
            inputs_embeds = torch.stack(all_embeds, dim=0).sum(dim=0)

        if self.dropout is not None:
            inputs_embeds = self.dropout(inputs_embeds)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        total_loss = None
        all_logits = []
        
        if labels is not None:
            total_loss = 0
            for i in range(self.config.num_channels):
                logits = self.heads[i](hidden_states)
                all_logits.append(logits)
                channel_labels = labels[:, :, i].view(-1)
                loss = self.criterions[i](logits.view(-1, logits.shape[-1]), channel_labels)
                total_loss += loss
        else:
            # In inference, we still want to compute all channel logits
            for i in range(self.config.num_channels):
                logits = self.heads[i](hidden_states)
                all_logits.append(logits)
        
        if not return_dict:
            output = (all_logits,) + outputs[1:]
            return (total_loss,) + output if total_loss is not None else output

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=all_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )