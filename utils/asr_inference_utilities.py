import os
import json
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Model,RWKV7Config
from transformers import AutoModelForCausalLM,AutoTokenizer
from model.llm.rwkv_asr import RWKV7ASRModel
import torch
chinese_instruction = "把以上音频转化为中文。 Assistant:"
english_instruction = "Convert the audios to English. Assistant:"

def load_asr_audio_lm_model(model_path):
    audio_lm_config_file = os.path.join(model_path, "audio_lm_model", "config.json")
    with open(audio_lm_config_file, "r") as f:
        audio_lm_config = json.load(f)
    rwkv_config = RWKV7Config.from_dict(audio_lm_config)
    audio_lm_model = RWKV7Model(rwkv_config)
    llm_model_path = os.path.join(model_path, "llm_model")
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path, trust_remote_code=True)
    asr_audio_lm_model = RWKV7ASRModel(audio_lm_model, llm_model)
    info = asr_audio_lm_model.load_state_dict(torch.load(os.path.join(model_path, "asr_audio_lm_model.pt")),strict=False)
    assert all(key.startswith("llm.") for key in info.missing_keys), "missing_keys are not started with 'llm.'"
    assert len(info.unexpected_keys) == 0, "unexpected_keys is not empty"
    print("✅loaded successfully")
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path,trust_remote_code=True)
    return asr_audio_lm_model,tokenizer

@torch.inference_mode()
def asr_inference_step(asr_audio_lm_model,tokenizer,semantic_tokens,language):
    print(f"semantic_tokens shape: {semantic_tokens.shape}")
    logits = asr_audio_lm_model.audio_lm_model(semantic_tokens,use_cache=True,return_dict=False)[0]
    print(f"logits shape before projection: {logits.shape}")
    logits = asr_audio_lm_model.projector(logits)
    print(f"logits shape after projection: {logits.shape}")
    if language == "chinese":
        instruction = chinese_instruction
    else:
        instruction = english_instruction
    text_input_ids = tokenizer.encode(instruction)
    text_input_ids = torch.tensor([text_input_ids],dtype=torch.long).to(semantic_tokens.device)
    print(f"text_input_ids: {text_input_ids}")
    print(f"text_input_ids shape: {text_input_ids.shape}")
    text_embeds = asr_audio_lm_model.llm.get_input_embeddings()(text_input_ids)
    print(f"text_embeds shape: {text_embeds.shape}")
    inputs_embeds = torch.cat([logits,text_embeds],dim=1)
    print(f"inputs_embeds shape: {inputs_embeds.shape}")
    attention_mask = torch.ones(inputs_embeds.shape[0],inputs_embeds.shape[1],dtype=torch.bool).to(semantic_tokens.device)
    print(f"attention_mask shape: {attention_mask.shape}")
    gen_args = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "max_length": 512,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "eos_token_id": 0
    }
    outputs = asr_audio_lm_model.llm.generate(**gen_args)
    print(f"outputs : {outputs}")
    print(f"outputs shape: {outputs.shape}")
    outputs = outputs[0][:-1].tolist()
    print(tokenizer.decode(outputs))
    return outputs

if __name__ == "__main__":
    model_path = "/home/yueyulin/models/rwkv7_0.1b_audio_lm_75k"
    asr_audio_lm_model,tokenizer = load_asr_audio_lm_model(model_path)
    print(asr_audio_lm_model)
    print(tokenizer)
    print(asr_audio_lm_model.audio_lm_model)
    print(asr_audio_lm_model.projector)
    print(asr_audio_lm_model.llm)
    asr_audio_lm_model.eval()
    asr_audio_lm_model = asr_audio_lm_model.to(torch.bfloat16).to("cuda")
    semantic_tokens = [5668, 4996, 7630, 3136, 7358, 4764, 856, 7627, 3061, 664, 3328, 852, 4103, 2958, 6932, 4611, 2184, 228, 7163, 316, 7746, 3951, 3952, 5058, 2288, 5881, 4212, 1204, 5853, 2066, 4644, 3835, 1338, 541, 6264, 5090, 3199, 7088, 3134, 6221, 4643, 5041, 4788, 7400, 1004, 813, 1011, 8117, 8028, 2166, 477, 55, 2294, 978, 1214, 2085, 7140, 7549, 1969, 7573, 3055, 4273, 1368, 4298, 1342, 6120, 1616, 1993, 4743, 6835, 3689, 1595, 6068, 3632, 460, 5437, 6408, 6202, 5240, 5620, 5572, 5553, 4686, 2436, 2013, 6663, 5765, 2293, 1717, 1631, 457, 4097, 1045, 7162, 453, 7895, 3513, 227, 6073, 5166, 56, 132, 422, 6896, 7236, 3110, 4424, 1545, 5236, 2119, 3137, 3114, 1877, 541, 836, 3858, 7116, 8016, 1037, 3043, 4570, 1768, 2639, 6112, 6435, 7947, 1751, 5077, 3056, 5469, 5923, 227, 6254, 7675, 1761, 6007, 3094, 2456, 2534, 300, 4090, 3564, 2013, 7740, 4171, 5352, 6391, 120, 7376, 6143, 3463, 4670, 4716, 6702, 2417, 7977, 1422, 2828, 2897, 5553, 2164, 6345, 3496, 1935, 826, 5333, 5843, 6589, 2002, 2109, 6140, 3704, 6609, 6790, 4808, 2169, 6080, 687, 7932, 2851, 7035, 1951, 3606, 7863, 4780, 5355, 4823, 4798, 3091, 2581, 715, 6018, 4657, 7921, 6413, 4298, 6176, 6499, 3856, 7293, 7074, 3416, 4105, 4452, 6097, 6708, 3560, 4475, 4047, 6437, 5223, 6696, 4769, 6643, 1504, 7821, 7631, 7475]
    semantic_tokens = torch.tensor([semantic_tokens],dtype=torch.long).to("cuda")
    logits = asr_inference_step(asr_audio_lm_model,tokenizer,semantic_tokens,"chinese")
    print(logits)