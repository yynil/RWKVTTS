import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Dict, Unpack
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils.deprecation import deprecate_kwarg
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Model, RWKV7PreTrainedModel, Cache,RWKV7ForCausalLM
from rwkvfla.models.rwkv7.modeling_rwkv7 import FusedLinearCrossEntropyLoss, FusedCrossEntropyLoss
from transformers.generation.utils import GenerationMixin

from rwkvfla.models.rwkv7.configuration_rwkv7 import RWKV7Config

class RWKV7SpeechConfig(RWKV7Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_vocab_size = kwargs.get("text_vocab_size", kwargs.get("text_vocab_size"))
        self.audio_global_vocab_size = kwargs.get("audio_global_vocab_size", kwargs.get("audio_global_vocab_size"))


class RWKV7ForSpeech(RWKV7ForCausalLM):
    config_class = RWKV7SpeechConfig
    def __init__(self, config: RWKV7SpeechConfig):
        super().__init__(config)
        self.model = RWKV7Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)#Spark 0.5B vocab size is 8192 + 1 for eos resulting in 8193
        self.criterion = None
        self.text_embedder = nn.Embedding(config.text_vocab_size, config.hidden_size)
        self.global_embedder = nn.Embedding(config.audio_global_vocab_size, config.hidden_size)#Spark 0.5B global token size is 4096
        #TTS Tag includes GLOBAL=0, SEMANTIC=1,START_TTS=2
        self.tts_tag_embedder = nn.Embedding(3, config.hidden_size)
        # Initialize weights and apply final processing
        self.post_init()
        self.dropout = torch.nn.Dropout(0.02)

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def generate(self, *args, **kwargs):
        try:
            return super().generate(*args, **kwargs)
        except AttributeError as exception:
            if 'past_key_values' in str(exception):
                raise AttributeError(
                    f"You tried to call `generate` with a decoding strategy that manipulates `past_key_values`, "
                    f"which is not supported for {self.__class__.__name__}. "
                    f"Try another generation strategy instead. "
                    f"For the available generation strategies, check this doc: "
                    f"https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies"
                )
            else:
                raise exception

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        logits_to_keep: Optional[int] = None,
        **kwargs
    ):
        # only last token for `inputs_ids` if the `past_key_values` is not empty.
        if past_key_values is not None and len(past_key_values) > 0:
            input_ids = input_ids[:, -1:]
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and len(past_key_values) == 0:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard.
            # Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {'input_ids': input_ids.contiguous()}

        if logits_to_keep is not None:
            model_inputs['logits_to_keep'] = logits_to_keep

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': use_cache,
            'attention_mask': attention_mask,
            'logits_to_keep': logits_to_keep,
        })
        return model_inputs

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
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
        logits_to_keep: Optional[int] = 0,
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.training and inputs_embeds is not None:
            inputs_embeds = self.dropout(inputs_embeds)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        hidden_states = outputs[0]
        fuse_linear_and_cross_entropy = self.config.fuse_cross_entropy and self.training

        loss, logits = None, None
        if not fuse_linear_and_cross_entropy or labels is None:
            logits = self.lm_head(hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:])
        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if fuse_linear_and_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss()
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            # Enable model parallelism
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if fuse_linear_and_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))

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
    
    def copy_state_dict(self, state_dict: dict):
        """从源 state dict 复制参数到当前模型，排除 embeddings 和 lm_head
        The state dict is from original RWKV7 language model
        Args:
            state_dict: 源 state dict
        """
        # 获取当前模型的 state dict
        target_dict = self.state_dict()
        
        # 创建新的 state dict 用于存储要复制的参数
        new_state_dict = {}
        
        # 遍历源 state dict 的键
        for key in state_dict.keys():
            # 跳过 embeddings 和 lm_head 相关的参数
            if key == 'model.embeddings.weight':
                new_state_dict['text_embedder.weight'] = state_dict[key]
                continue
            if 'embeddings' in key or 'lm_head' in key:
                continue
            # 如果键在当前模型中存在，则复制参数
            if key in target_dict:
                new_state_dict[key] = state_dict[key]
        
        # 加载新的 state dict 到当前模型
        info = self.load_state_dict(new_state_dict, strict=False)
        print(info)
        return self
        