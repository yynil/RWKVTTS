import torch
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from langdetect import detect
from utils.properties_util import classify_pitch,classify_speed
from utils.phonem_utils import ramdomly_mark_phonem_natural_tagged
import random
def detect_language(text):
    try:        
        if detect(text).startswith('zh'):
            return 'zh'
        else:
            return 'en'
    except:
        return 'en'

age_description_map = {
    "child": "child",
    "teenager": "teenager",
    "youth-adult": "youth",
    "middle-aged": "middle-aged",
    "elderly": "elderly",
}
gender_description_map = {
    "female": "female",
    "male": "male",
}
emotion_description_map = {
    "UNKNOWN": "unknown",
    "NEUTRAL": "neutral",
    "ANGRY": "angry",
    "HAPPY": "happy",
    "SAD": "sad",
    "FEARFUL": "fearful",
    "DISGUSTED": "disgusted",
    "SURPRISED": "surprised",
    "SARCASTIC": "sarcastic",
    "EXCITED": "excited",
    "SLEEPY": "sleepy",
    "CONFUSED": "confused",
    "EMPHASIS": "emphasis",
    "LAUGHING": "laughing",
    "SINGING": "singing",
    "WORRIED": "worried",
    "WHISPER": "whisper", 
    "ANXIOUS": "anxious",
    "NO-AGREEMENT": "no-agreement",
    "APOLOGETIC": "apologetic",
    "CONCERNED": "concerned",
    "ENUNCIATED": "enunciated",
    "ASSERTIVE": "assertive",
    "ENCOURAGING": "encouraging",
    "CONTEMPT": "contempt",
}

speed_description_map = {
    "very_slow": "very slow",
    "slow": "slow",
    "medium": "medium",
    "fast": "fast",
    "very_fast": "very fast",
}

pitch_description_map = {
    "low_pitch": "low",
    "medium_pitch": "medium",
    "high_pitch": "high",
    "very_high_pitch": "very high",
}
INSTRUCTION = "User: Please generate the speech according to the following text: {text}\nAssistant:"
INSTRUCTION_WITH_PROPERTIES = "User: Please generate the speech with the properties: {properties} according to the following text: {text}\nAssistant:"

def create_properties_description(batch):
    age = age_description_map[batch['age'].lower()]
    gender = gender_description_map[batch['gender'].lower()]
    emotion = emotion_description_map[batch['emotion']]
    pitch = batch['pitch']
    speed = batch['speed']
    pitch_str = pitch_description_map[classify_pitch(pitch, gender, age)]
    speed_str = speed_description_map[classify_speed(speed)]
    return f"Age: {age} Gender: {gender} Emotion: {emotion} Pitch: {pitch_str} Speed: {speed_str}"


def create_inputs_and_labels(batch, tokenizer,eos_id=8192, is_global_tokens_predictable=False, device="cpu",is_properties_used=False,randomly_mark_phonems=False,random_mark_prob=0.5):
    """创建TTS模型的输入和标签"""
    # 这里需要根据你的数据格式来实现
    # 假设batch包含：text, global_tokens, semantic_tokens
    bsz = len(batch)
    text_input_ids_list = []
    text_attention_mask_list = []
    audio_token_ids_list = []
    audio_token_attention_mask_list = []
    labels_list = []
    for i in range(bsz):
        text = batch[i]['text']
        global_tokens = batch[i]['global_tokens']
        semantic_tokens = batch[i]['semantic_tokens']
        if randomly_mark_phonems:
            if random.random() < random_mark_prob:
                text = ramdomly_mark_phonem_natural_tagged(text,lang=detect_language(text),min_mark=1,max_mark=3,wrong_word_ratio=0.5)
        if is_properties_used:
            properties_str = create_properties_description(batch[i])
            instruction = INSTRUCTION_WITH_PROPERTIES.format(properties=properties_str,text=text)
        else:
            instruction = INSTRUCTION.format(text=text)
        input_ids = tokenizer.encode(instruction)
        text_input_ids_list.append(torch.tensor(input_ids,dtype=torch.long,device=device))
        text_attention_mask_list.append(torch.ones(len(input_ids),dtype=torch.long,device=device))

        global_tokens = [x + 8193 for x in global_tokens]
        audio_token_ids = global_tokens + semantic_tokens + [eos_id]
        audio_token_ids_list.append(torch.tensor(audio_token_ids,dtype=torch.long,device=device))
        audio_token_attention_mask_list.append(torch.ones(len(audio_token_ids),dtype=torch.long,device=device))
        label_tensor = torch.tensor(audio_token_ids,dtype=torch.long,device=device)
        if not is_global_tokens_predictable:
            label_tensor[0:len(global_tokens)] = -100
        labels_list.append(label_tensor)


    return {
        "text_input_ids": pad_sequence(text_input_ids_list,batch_first=True,padding_value=0,padding_side="left"),
        "text_attention_mask": pad_sequence(text_attention_mask_list,batch_first=True,padding_value=0,padding_side="left"),
        "audio_token_ids": pad_sequence(audio_token_ids_list,batch_first=True,padding_value=0,padding_side="left"),
        "audio_token_attention_mask": pad_sequence(audio_token_attention_mask_list,batch_first=True,padding_value=0,padding_side="left"),
        "labels": pad_sequence(labels_list,batch_first=True,padding_value=-100,padding_side="left")
    }


if __name__ == "__main__":
    model_path = "/home/yueyulin/models/rwkv7-0.4B-g1"
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    from datasets import load_dataset
    import glob
    import os
    jsonl_dir = "/home/yueyulin/data/voxbox_wids_tokens_filtered/ncssd_c_en/"
    jsonl_files = glob.glob(os.path.join(jsonl_dir, "*.jsonl"))
    dataset = load_dataset("json", data_files=jsonl_files)['train']
    print(dataset)
    output = tokenizer(["你好","世界很大，我想去看看"],padding=True,padding_side="left",
        return_tensors="pt")
    print(output)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset,batch_size=4,shuffle=True,collate_fn=lambda x: x)
    for batch in dataloader:
        print(batch) 
        global_predictable_batch = create_inputs_and_labels(batch, tokenizer, is_global_tokens_predictable=True, device="cuda:0",is_properties_used=True)
        print(global_predictable_batch)
        print(tokenizer.decode(global_predictable_batch['text_input_ids'][0],skip_special_tokens=True))
        print(tokenizer.decode(global_predictable_batch['text_input_ids'][1],skip_special_tokens=True))
        print('-'*100)
        randomly_mark_phonems_batch = create_inputs_and_labels(batch, tokenizer, device="cuda:0",randomly_mark_phonems=True,random_mark_prob=0.5,is_global_tokens_predictable=False,is_properties_used=False)
        print(randomly_mark_phonems_batch)
        print(tokenizer.decode(randomly_mark_phonems_batch['text_input_ids'][0],skip_special_tokens=True))
        print(tokenizer.decode(randomly_mark_phonems_batch['text_input_ids'][1],skip_special_tokens=True))
        print(tokenizer.decode(randomly_mark_phonems_batch['text_input_ids'][2],skip_special_tokens=True))
        print(tokenizer.decode(randomly_mark_phonems_batch['text_input_ids'][3],skip_special_tokens=True))
        print('-'*100)
        break