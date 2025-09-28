import torch
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from pathlib import Path
import re
import soundfile as sf
def spark_inference(language_model, audio_tokenizer, text_tokenizer, text,global_tokens_ids,device:str,eos_token_id):
    """
    Args:
        language_model: 
        audio_tokenizer: 
        text_tokenizer: 
        text: 
    """
    language_model.eval()

    model_inputs = text_tokenizer(text, return_tensors="pt").to(device)
    print(model_inputs)
    len_of_input = model_inputs['input_ids'].shape[1]
    with torch.no_grad():

        generated_tokens = language_model.generate(
            **model_inputs,
            max_new_tokens=3000,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            eos_token_id=eos_token_id,
        )[0]
    generated_tokens = generated_tokens[len_of_input:]
    predicts = text_tokenizer.decode(generated_tokens, skip_special_tokens=True)
    pred_semantic_ids = torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicts)])
    pred_semantic_ids = pred_semantic_ids.long().unsqueeze(0)
    pred_semantic_ids = pred_semantic_ids.to(device)
    print(generated_tokens)
    print(pred_semantic_ids)
    print("--------------------------------")
    global_tokens_ids = torch.tensor(global_tokens_ids,dtype=torch.long).unsqueeze(0).to(device)
    print(f'global_tokens: {global_tokens_ids.shape}, semantic_ids: {pred_semantic_ids.shape}')
    with torch.no_grad():
        wav = audio_tokenizer.detokenize(global_tokens_ids,pred_semantic_ids)
    return wav

def load_global_tokens(audio_tokenizer, directory,device:str):
    """
    Args:
        audio_tokenizer: 
        directory: 
    """
    ref_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                ref_files.append(Path(os.path.join(root, file)))
   
    print(f'load global tokens from {ref_files}')
    all_global_tokens = []
    all_global_tokens_ids = []
    characters = []
    for ref_file in ref_files:
        parent_dir = os.path.dirname(ref_file)
        global_tokens_ids, semantic_tokens = audio_tokenizer.tokenize(ref_file)
        global_tokens_ids = global_tokens_ids.squeeze(0).squeeze(0).detach().cpu().tolist()
        global_tokens ="".join([f"<|bicodec_global_{token}|>" for token in global_tokens_ids])
        all_global_tokens.append(global_tokens)
        all_global_tokens_ids.append(global_tokens_ids)
        characters.append(parent_dir.split('/')[-1])
    return all_global_tokens, all_global_tokens_ids,characters

if __name__ == "__main__":
    model_dir = '/home/yueyulin/models/Spark-TTS-0.5B/'
    demo_dir = '/home/yueyulin/github/RWKVTTS/demos/'
    device = 'cpu'
    audio_tokenizer = BiCodecTokenizer(model_dir, device=device)
    text_tokenizer = AutoTokenizer.from_pretrained(model_dir+"LLM")
    language_model = AutoModelForCausalLM.from_pretrained(model_dir+"LLM").to(device)
    all_global_tokens,all_global_tokens_ids,characters = load_global_tokens(audio_tokenizer, demo_dir,device)
    text = "你好，我是小明，我是一个学生，我正在学习英语。"
    index = 0
    eos_token_id = text_tokenizer.eos_token_id
    print(f'eos_token_id: {eos_token_id}')
    for global_tokens,global_tokens_ids,character in zip(all_global_tokens,all_global_tokens_ids,characters):
        print(global_tokens)
        inputs = [
                "<|task_tts|>",
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
            ]
        input_text = "".join(inputs)
        wav = spark_inference(language_model, audio_tokenizer, text_tokenizer, input_text,global_tokens_ids,device,eos_token_id)
        sf.write(f"from_spark_audio_tokens_{character}.wav", wav, audio_tokenizer.config['sample_rate'])
        print(f"save wav to from_spark_audio_tokens_{character}.wav")
    # text = 'Hello, how are you?'
    # inputs = [
    #             "<|task_tts|>",
    #             "<|start_content|>",
    #             text,
    #             "<|end_content|>",
    #             "<|start_global_token|>",
    #             global_tokens,
    #             "<|end_global_token|>",
    #         ]

