from operator import is_
from click import Option, prompt
from numpy import dtype
from pydantic import InstanceOf
import torch
from torch import nn
from typing import List, Callable,Dict, Optional,Generator,AnyStr,Union

import transformers
from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
from cosyvoice.utils.common import IGNORE_ID
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
import torch.nn.functional as F
from cosyvoice.utils.common import th_accuracy
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
import time
class RWKV7LM(nn.Module):
    def __init__(
            self,
            llm_input_size: int,
            llm_output_size: int,
            speech_token_size: int,
            llm: Union[AutoModelForCausalLM,AnyStr],
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            mix_ratio: List[int] = [5, 15],
            drop_ratio = 0.0,
            vocab_size = 0,
    ):
        super(RWKV7LM, self).__init__()
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2

        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        if isinstance(llm, str):
            #load configuration and init model withouth loading weights
            model_configuration = AutoConfig.from_pretrained(llm,trust_remote_code=True)
            self.llm = AutoModelForCausalLM.from_config(model_configuration,trust_remote_code=True)
            if vocab_size != 0:
                from train_scripts.train_functions import alter_emb_and_head # Only used for inference
                self.llm = alter_emb_and_head(self.llm,vocab_size,speech_token_size)
        else:
            self.llm = llm
        self.text_embedding = self.llm.get_input_embeddings()
        # self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 1, llm_input_size)

        # 4. sampling method
        self.sampling = sampling
        self.mix_ratio = mix_ratio
        
        #Dropout
        if drop_ratio > 0:
            self.dropout = nn.Dropout(drop_ratio)
        else:
            self.dropout = None
        
    def pad_unpad_sequence(self, sos_eos_emb, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        device = text_token.device
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_eos_emb.squeeze(dim=0),  text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        # lm_input_len = [i.size(0) for i in lm_input]
        attention_mask = [torch.ones(i.size(0),device=device,dtype=torch.int32) for i in lm_input]
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        return lm_input, attention_mask
    def forward(
            self,
            batch: dict,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token']
        text_token_len = batch['text_token_len']
        speech_token = batch['speech_token']
        speech_token_len = batch['speech_token_len']

        # 1. prepare llm_target
        lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token[i, :speech_token_len[i]].tolist() +
                                  [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(text_token.device)

        # 1. encode text_token
        text_token = self.text_embedding(text_token)
        


        # 3. eos and task_id
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 5. unpad and pad
        lm_input, attention_mask = self.pad_unpad_sequence(sos_eos_emb, text_token, text_token_len,
                                                         task_id_emb, speech_token, speech_token_len)
        # 5.1 create attention mask
        # attention mask is [1,text_token_len,1,sp_token_len,0,0,0]
        
        if self.dropout is not None:
            lm_input = self.dropout(lm_input)
        # 6. run lm forward
        lm_output = self.llm(inputs_embeds=lm_input, attention_mask=attention_mask,output_hidden_states=True,return_dict=True)
        # hidden_states = lm_output.hidden_states[-1]
        # logits = self.llm_decoder(hidden_states)
        logits = lm_output.logits
        lm_target = lm_target[:, 1:].contiguous()
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 1), lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}
    
    def dummy_forward(self):
        print(f'start to do dummy forward')
        with torch.no_grad():
            with torch.amp.autocast(enabled=True,device_type='cuda'):
                xs = torch.ones(1, 1, self.llm_input_size,device=self.llm.model.device,dtype=torch.float)
                print(f'xs is {xs.dtype}')
                masks = torch.ones(1, 1, 1,device=self.llm.model.device,dtype=torch.long)
                cache = None
                self.forward_one_step(xs, masks, cache)
        print(f'finish dummy forward')
    
    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.llm(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        logits = outs.logits
        new_cache = outs.past_key_values
        return logits, new_cache
    
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
    @torch.inference_mode
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
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        
        end_of_prompt_id = 65531
        #find the length of instruction and text the text is [prompt, end_of_prompt, text]
        end_of_prompt_mask = (text == end_of_prompt_id)
        # 使用nonzero找到所有匹配的索引
        end_of_prompt_indices = end_of_prompt_mask.nonzero()
        
        # 默认值：没有找到end_of_prompt_id
        instruction_length = 0
        content_length = text_len
        
        # 如果找到了end_of_prompt_id
        if end_of_prompt_indices.size(0) > 0:
            # 获取第一个匹配的索引（只考虑第一个出现的end_of_prompt_id）
            # 由于text是二维张量 [batch, seq_len]，我们需要第二个维度的索引
            instruction_length = end_of_prompt_indices[0, 1].item()
            content_length = text_len - (instruction_length + 1)  # +1是因为要跳过end_of_prompt_id标记本身
            # print(f'找到end_of_prompt标记，指令长度: {instruction_length}, 内容长度: {content_length}')
    
        text_len += prompt_text_len
        text = self.text_embedding(text)
        

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = content_length * min_token_text_ratio
        max_len = content_length * max_token_text_ratio
        # print(f'min_len is {min_len}, max_len is {max_len}')
        # 5. step by step decode
        out_tokens = []
        cache = None
        start_time = time.time()
        end_time = 0
        is_prefill = True
        prefill_time = 0
        prefill_length = lm_input.shape[1]
        for i in range(max_len):
            logits, cache = self.forward_one_step(lm_input,
                                                      masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                                      cache=cache)
            # print(f'logits.shap is {logits.shape}')
            logp = logits[:,-1].log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            if top_ids > self.speech_token_size:
                continue
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
            if is_prefill:
                prefill_time = time.time() - start_time
                is_prefill = False
        end_time = time.time()
        decode_time = end_time - start_time - prefill_time
        decoded_length = len(out_tokens)
        print(f'tps for prefill is {prefill_length/prefill_time}. {prefill_length} tokens in {prefill_time} seconds')
        print(f'tps for decode is {decoded_length/decode_time}. {decoded_length} tokens in {decode_time} seconds')
        print(f'out_tokens is {out_tokens}')
    
if __name__ == '__main__':
    rwkv_path = "/external_data/models/rwkv7-191M-world/"
    json_data = {"text": "馬克思去世之後，恩格斯在1884年所出版的《家庭、私有制和國家的起源》，被視為馬克思主義民族學的第一本經典著作。提到了民族形成的規律。人結合成群；由血緣關係組成原始的家庭型式並形成親屬制度，逐漸排除同胞的性交關係；共祖的血族團體結成氏族，氏族結成部落，進而結成部落聯盟，融合成「人民（）」；隨著生產力的增加，分工擴大，新的生產關係出現，新階級產生，使得氏族制度漸漸不能負荷而消滅，隨之產生由「新國族（）」組成的國家。", "tts_speech_tokens": [1959, 1707, 1704, 5835, 5832, 5832, 3645, 3888, 2031, 2112, 2124, 4133, 5672, 489, 2268, 6453, 5643, 4527, 159, 6298, 3810, 4612, 2989, 4680, 3719, 3477, 4671, 4610, 2685, 4953, 5080, 2897, 2087, 1978, 1948, 1950, 2112, 1028, 2, 6175, 507, 2585, 4511, 5725, 6374, 3465, 5177, 3718, 6375, 2261, 5189, 35, 1763, 2015, 5667, 4777, 2166, 703, 6077, 649, 2163, 5672, 6076, 4466, 5644, 6374, 6372, 4527, 753, 2699, 341, 86, 2234, 4995, 5808, 725, 710, 632, 2085, 6213, 3321, 5830, 2910, 720, 3890, 1463, 1476, 1746, 59, 353, 2540, 1514, 1951, 2031, 4920, 4758, 2276, 6320, 5914, 222, 2193, 2688, 3509, 5207, 5392, 5644, 5645, 4680, 4615, 1127, 725, 1370, 3462, 4915, 3465, 803, 632, 2754, 576, 623, 2810, 2909, 1410, 663, 1457, 2864, 231, 2509, 5921, 5914, 54, 4523, 4920, 4596, 5298, 3840, 1943, 1691, 734, 1460, 1800, 1804, 2031, 2112, 4299, 2160, 2378, 2816, 2267, 5400, 5645, 5644, 4680, 4609, 2185, 1454, 224, 2582, 4526, 1762, 3651, 5835, 3648, 1458, 2031, 2121, 4311, 4133, 6401, 594, 324, 6454, 5643, 4527, 3312, 3462, 4032, 1454, 4046, 5588, 5749, 2753, 1884, 1885, 512, 4480, 5529, 3228, 4041, 5825, 5179, 6055, 1635, 137, 135, 64, 2246, 1896, 60, 5669, 5668, 5100, 954, 2166, 380, 8, 1725, 1284, 4759, 3562, 4560, 6018, 2482, 440, 737, 1483, 6216, 2178, 4289, 3963, 5493, 4370, 707, 626, 1247, 2112, 4299, 4218, 3894, 3645, 1701, 4218, 2112, 60, 4461, 565, 2419, 54, 5023, 4949, 3151, 2432, 1153, 1884, 1079, 4723, 6258, 5412, 4042, 6557, 4453, 4596, 528, 2456, 6177, 4672, 748, 909, 4394, 712, 573, 696, 5825, 4615, 240, 4614, 6070, 1700, 1843, 1735, 3969, 3645, 1461, 2031, 1611, 1688, 113, 5208, 3993, 4678, 1295, 467, 2405, 4860, 2792, 2324, 2733, 2214, 752, 2213, 6100, 6055, 6024, 1672, 395, 3670, 1726, 1950, 3648, 5835, 5832, 3651, 2031, 1266, 3481, 968, 707, 5098, 5581, 5971, 6055, 942, 962, 953, 2357, 4546, 5773, 654, 5090, 5996, 4543, 4723, 4512, 4759, 4659, 5020, 1052, 1534, 1734, 5484, 2097, 1457, 2099, 4671, 2187, 740, 17, 5453, 4129, 1941, 1700, 815, 2195, 4557, 5409, 5402, 5401, 1044, 792, 306, 110, 151, 4920, 2347, 6158, 6157, 141, 2193, 3428, 5696, 4659, 4516, 528, 539, 2942, 5166, 5645, 4914, 4528, 788, 1735, 1950, 5832, 5832, 3645, 1788, 2112, 1920, 5344, 4966, 4886, 4444, 4596, 1267, 4481, 5367, 4672, 748, 5, 4479, 3912, 4675, 4488, 2724, 4560, 5148, 6534, 2169, 2102, 1854, 4920, 3231, 5177, 1125, 303, 2180, 1457, 2096, 2113, 4134, 3888, 3645, 1701, 2031, 5649, 4113, 2186, 3153, 4920, 4849, 5192, 2125, 2175, 6075, 1460, 3653, 2418, 5484, 4320, 2105, 5831, 3312, 1032, 2430, 512, 1403, 6501, 2181, 703, 6404, 5030, 5077, 144, 73, 59, 4435, 4594, 4569, 4966, 3994, 2499, 962, 2840, 717, 672, 1379, 3653, 4643, 4443, 4756, 4686, 3239, 1771, 1708, 4218, 4218, 6405, 6405, 4218, 4218, 2112, 570, 5097, 5047, 5774, 3991, 6378, 4041, 2186, 1454, 300, 56, 2324, 2297, 785, 1978, 2031, 1734, 5217, 6052, 3867, 971, 221, 65, 1284, 6222, 2183, 5826, 948, 2511, 6328, 3650, 5289, 1041, 4380, 1266, 2024, 1356, 4677, 1286, 4487, 3459, 2466, 2201, 5203, 4680, 4996, 4753, 4527, 5420, 3468, 3378, 2180, 1457, 2095, 2059, 4218, 4218, 3894, 4482, 4753, 4752, 5418, 5499, 3468, 2099, 5831, 2424, 4677, 395, 4517, 4680, 2233, 3340, 6258, 2166, 1431, 5829, 5099, 1692, 725, 710, 1355, 1757, 2032, 4218, 4218, 2031, 2733, 4515, 5075, 4562, 335, 4052, 5324, 384, 4678, 1286, 2249, 2332, 4915, 2925, 983, 4068, 2157, 2166, 2901, 5089, 3879, 707, 380, 75, 638, 737, 2610, 6501, 671, 581, 2777, 1238, 1960, 2031, 4218, 3894, 5838, 3651, 3645, 1701, 1947, 1854, 1693, 1445, 5047, 5046, 498, 2430, 2819, 4727, 147, 5644, 2925, 731, 2210, 5938, 1744, 1584, 1685, 86, 6255, 6261, 4071, 2726, 2726, 3751, 2049, 1959, 4218, 4218, 4218, 4218, 4218, 4137, 5454, 5805, 883, 224, 2428, 5649, 6309, 5828, 1454, 5157, 5644, 4412, 4554, 6099, 4428, 3655, 1463, 1588, 2564, 4858, 792, 56, 64, 5650, 2503, 4415, 6018, 3453, 4686, 2573, 5915, 6238, 1703, 1705, 1950, 6345, 6534, 4347, 2810, 6175, 6181, 588, 5075, 5775, 3589, 651, 4947, 5074, 4970, 54, 2189, 6077, 1703, 1948, 1947, 1704, 3645, 3651, 1950, 4404, 4513, 2571, 4993, 6018, 4419, 56, 5644, 2457, 4394, 5934, 5652, 2930, 1460, 2047, 699, 3566, 4544, 4659, 2580, 3724, 5649, 4077, 3616, 5342, 4515, 2553, 4841, 737, 1493, 2032, 4218, 6405, 4218, 4218, 2031, 4650, 4756, 4515, 5668, 5452, 6261, 4929, 4686, 4607, 4597, 393, 4677, 2024, 5182, 5649, 2187, 3655, 1466, 752, 5656, 5643, 56, 2225, 5203, 1744, 2031, 4137, 5832, 1701, 3894, 4644, 5402, 3222, 1532, 315, 2324, 4834, 4671, 5726, 4689, 5499, 4197, 4286, 5099, 1125, 4998, 4680, 4528, 1851, 396, 3644, 1370, 1275, 4434, 4849, 4472, 3426, 4669, 4571, 4311, 2175, 2898, 2655, 991, 119, 5042, 4257, 6534, 6534, 5102, 2831, 2673, 495, 632, 2891, 2783, 2006, 1951, 4218, 4137, 2027, 2108, 1865, 3863, 2336, 4593, 4839, 2393, 5756, 4311, 4562, 2528, 749, 5724, 4077, 1700, 4615, 6378, 5661, 5256, 5655, 2214, 3656, 737, 752, 5412, 5645, 74, 2216, 4474, 1725, 1704, 5832, 3645, 2031, 1950, 1132, 1937, 5095, 4435, 4594, 2553, 4831, 5532, 6505, 672, 728, 5054, 2413, 5406, 4286, 5831, 3312, 6213, 2183, 2099, 4752, 11, 1557, 4385, 2899, 1383, 1457, 719, 4592, 2733, 4697, 5273, 6157, 5428, 1730, 1975, 1701, 5832, 3645, 1461, 4299, 2112, 135, 5915, 6157], "prompt_text": "他和之前的那位护士粉丝一样呀，都很喜欢心理咨询。", "llm_prompt_speech_token": [2058, 5103, 6076, 5996, 3195, 2440, 4807, 72, 5727, 5658, 5175, 5260, 4755, 2598, 3659, 181, 2225, 4492, 1482, 4403, 3609, 3098, 2663, 2834, 470, 2850, 2112, 2913, 5095, 5403, 36, 2169, 4329, 902, 749, 4194, 6374, 46, 5020, 3454, 3480, 1295, 197, 6481, 5752, 734, 737, 758, 1954, 1957, 1954, 2058, 1842, 74, 716, 5831, 5101, 2676, 2268, 1712, 818, 5453, 5371, 4755, 4759, 4758, 1294, 1847, 2769, 584, 3971, 4472, 4660, 4839, 2562, 4805, 3751, 998, 53, 4194, 6374, 6454, 6390, 5257, 4443, 6298, 1671, 953, 1582, 2041]}
    json_data_2 = {"text": "早期的制度主义者受到传统政治哲学和欧洲大陆国家学影响，主要关注自上而下的制度设计问题。认为制度是影响人类行为的基本因素，对政治机构运作的研究主要通过对政治制度中权力分配和人类行为的法律与机构约束地了解。主要通过制度研究法和历史比较法进行研究。", "tts_speech_tokens": [4131, 6075, 5832, 5862, 3645, 1701, 1950, 4299, 4920, 2259, 3404, 3410, 2786, 156, 5325, 2283, 150, 4450, 4430, 1029, 4918, 4752, 4770, 1281, 5743, 5015, 1369, 5727, 2180, 1697, 5588, 4614, 3720, 4671, 884, 308, 542, 4644, 4995, 5068, 5095, 552, 4646, 2756, 387, 5487, 3321, 1955, 1703, 4554, 3993, 2763, 692, 1967, 1887, 4920, 4761, 4608, 5122, 1805, 2022, 4995, 5661, 6149, 1774, 3462, 4915, 794, 632, 4603, 6027, 3840, 1700, 233, 137, 5427, 495, 1280, 3476, 5015, 5098, 5014, 3462, 2826, 2912, 5014, 2139, 300, 5509, 6319, 2837, 3879, 5825, 3641, 2186, 1392, 1457, 716, 150, 2337, 3086, 6401, 2408, 5325, 6270, 1672, 962, 2267, 3482, 1268, 3455, 4830, 4416, 4569, 2276, 1460, 1946, 1702, 1950, 6075, 5832, 5859, 5862, 5835, 3645, 1944, 4218, 1977, 4833, 1855, 1616, 4607, 5509, 5054, 1456, 654, 2913, 5027, 5104, 5931, 3480, 3384, 5824, 4289, 1370, 1923, 2490, 4753, 5013, 5742, 6148, 4672, 4914, 4456, 5357, 1235, 1703, 1784, 3812, 4526, 4593, 2301, 4543, 6320, 55, 315, 272, 299, 1758, 6351, 4673, 4752, 4770, 1368, 306, 5825, 1454, 5238, 4996, 2790, 4753, 148, 3228, 3158, 3637, 5830, 2324, 1561, 555, 4488, 294, 1295, 800, 1762, 1750, 6147, 4447, 4780, 5777, 1457, 227, 224, 4850, 4534, 2032, 1950, 5838, 5835, 5862, 5859, 3888, 4137, 6405, 6405, 4218, 5379, 4753, 4923, 4851, 2504, 2094, 306, 5095, 5828, 1457, 2096, 2086, 1975, 5400, 4753, 4672, 4932, 5420, 809, 1295, 2024, 5642, 5073, 4659, 4839, 4463, 4133, 1226, 1748, 3474, 803, 38, 5853, 6093, 4486, 4598, 314, 2510, 4488, 4731, 4812, 1106, 1883, 5776, 5825, 2348, 1041, 2243, 58, 1248, 4512, 2472, 4857, 1686, 3627, 167, 269, 1268, 3536, 5480, 5933, 5238, 5562, 3636, 725, 1364, 2093, 2112, 1113, 4609, 2420, 160, 2139, 5647, 4761, 4599, 5121, 3994, 5652, 3961, 1041, 4758, 3319, 2428, 1227, 5015, 2909, 4129, 6073, 4587, 5938, 1807, 5484, 5824, 4997, 1044, 2252, 1052, 1294, 1133, 8, 3022, 3426, 2589, 5342, 5090, 387, 4191, 6294, 1828, 2099, 3158, 4568, 5591, 307, 300, 5598, 5048, 5776, 591, 2913, 5084, 542, 1923, 381, 4601, 2663, 80, 1599, 1977, 4753, 4581, 4518, 3666, 3913, 4680, 6392, 3231, 3462, 4996, 4761, 5094, 1044, 1038, 5096, 3560, 1119, 5490, 2891, 2861, 5774, 3991, 1951, 4218, 5865, 6052, 3084, 818, 1463, 3825, 4561, 5020, 1684, 6534, 2530, 4511, 6255, 2130, 2628, 4570, 386, 1520, 1732, 5346, 4698, 1261, 632, 2576, 3477, 1774, 785, 2934, 6177, 4420, 4570, 638, 4435, 4542, 357, 1673, 6506, 3395, 4778, 876, 2260, 29, 1028, 6507, 6535, 5805, 3491, 1946, 1703, 839, 1533, 4613, 5344, 4130, 1943, 3239, 879, 4920, 4758, 5668, 4858, 1218, 4852, 5084, 638, 2087, 1945, 3645, 5832, 5859, 5859, 3675, 3648, 4137, 6405, 4218, 3885, 6316, 6074, 4598, 4573, 4671, 5076, 5562, 5095, 4373, 2186, 2012, 2031, 2112, 60, 4776, 5019, 4939, 1295, 1772, 1762, 1707, 4218, 2031, 80, 314, 1784, 3491, 4843, 312, 2337, 2672, 386, 791, 1493, 1708, 4137, 4218, 6405, 6405, 6324, 6324, 4920, 5562, 2180, 1856, 3320, 4594, 5674, 2702, 1839, 351, 4875, 5039, 5776, 5532, 1401, 669, 5091, 5003, 515, 64, 2004, 4753, 4770, 5013, 2091, 2503, 5099, 1697, 1376, 386, 737, 4479, 3264, 3066, 6071, 5089, 3430, 6535, 604, 1946, 1946, 1217, 4617, 3159, 623, 299, 3798, 4453, 5100, 4777, 2983, 4996, 4915, 2017, 1046, 1923, 1920, 1698, 1295, 1051, 2733, 4615, 5509, 2870, 6453, 6534, 1949, 1460, 222, 4677, 4533, 4832, 4407, 2454, 2726, 1268, 566, 116, 8, 6261, 2490, 3156, 2414, 5093, 5102, 2006, 2032, 1947, 3888, 5832, 3645, 3645], "prompt_text": "当然你的他对保证金有要求，有的是比如说百分之你交易值的百分之五。", "llm_prompt_speech_token": [1516, 1950, 4215, 1116, 4700, 4628, 5365, 4496, 2291, 1726, 1016, 886, 72, 4592, 1836, 2058, 2916, 6086, 4537, 1038, 4604, 5342, 2160, 2163, 2870, 1952, 29, 3471, 3476, 4718, 3751, 4686, 2310, 5588, 4831, 4777, 971, 956, 1295, 5749, 4568, 6323, 4868, 5077, 2257, 5325, 1623, 1607, 1862, 4130, 1199, 74, 1767, 4528, 4763, 5643, 4923, 6391, 1774, 2058, 1950, 1689, 1618, 1532, 5481, 5024, 2163, 712, 1463, 812, 2142, 5312, 5934, 5658, 4915, 4905, 3717, 1968, 3914, 5426, 1599, 1005, 5325, 4598, 3491, 4781, 4615, 4695, 5911, 3722, 1275, 4915, 4833, 2018, 1775, 3312, 306, 4538, 2178, 4299, 4347, 326, 1460, 1810, 3016, 6180, 4923, 6391, 4852, 5828, 2105, 2105, 2105, 728, 5098, 3551, 1951]}
    device = 'cuda'
    model = AutoModelForCausalLM.from_pretrained(rwkv_path, trust_remote_code=True).to(dtype=torch.bfloat16)
    model.to(device)
    configuration = model.config
    print(configuration)
    tokenizer = AutoTokenizer.from_pretrained(rwkv_path, trust_remote_code=True)
    print(tokenizer)    
    tokenizer.add_special_tokens({'pad_token': '<|rwkv_tokenizer_end_of_text|>'})
    
    
    llm_input_size = configuration.hidden_size
    llm_output_size = configuration.hidden_size
    speech_token_size = 6561
    rwkv7lm = RWKV7LM(llm_input_size, llm_output_size, speech_token_size, model,None).to(dtype=torch.bfloat16)
    rwkv7lm.to(device)
    rwkv7lm.train()
    print(rwkv7lm)
    
    speech_tokens = [torch.tensor(json_data["tts_speech_tokens"],dtype=torch.int32), torch.tensor(json_data_2["tts_speech_tokens"],dtype=torch.int32)]
    speech_length = torch.tensor([len(json_data["tts_speech_tokens"]), len(json_data_2["tts_speech_tokens"])],dtype=torch.int32)
    print(speech_length)

    speech_tokens = pad_sequence(speech_tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
    print(speech_tokens.shape)
    
    texts = [json_data["text"], json_data_2["text"]]
    prompts = [json_data["prompt_text"], json_data_2["prompt_text"]]
    texts_ids = [
        torch.tensor(tokenizer.encode(texts[i],add_special_tokens=False)+tokenizer.encode(prompts[i],add_special_tokens=False),dtype=torch.int32)
        for i in range(len(texts))
    ]
    texts_length = torch.tensor([i.shape[0] for i in texts_ids],dtype=torch.int32)
    print(texts_length)
    texts_ids = pad_sequence(texts_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    print(texts_ids.shape)
    
    
    
    batch = {"text_token": texts_ids.to(device), "text_token_len": texts_length.to(device), "speech_token": speech_tokens.to(device), "speech_token_len": speech_length.to(device)}
    output = rwkv7lm(batch)
    print(output)
    
    print(model)
    print(model.__class__.__name__)
    print(f"类名: {model.__class__.__name__}")
    print(f"完整类路径: {model.__class__.__module__}.{model.__class__.__qualname__}")