import click
import torch
from model.llm.rwkv_asr_whisper import RWKV7ASRModel,RWKV7ModelForLatentInputs,load_whisper_feature_extractor_and_encoder
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Config, RWKV7Model, RWKV7ForCausalLM
from tokenizer.rwkv_tokenizer import RWKV_TOKENIZER
import transformers
from transformers import AutoTokenizer,WhisperFeatureExtractor,WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder
import soundfile as sf
import os
@click.command()
@click.option("--audio_lm_path", type=str, required=True)
@click.option("--llm_path", type=str, required=True)
@click.option("--language", type=str, required=False,default="chinese")
@click.option("--device", type=str, required=False,default="cuda:0")
@click.option("--whisper_path", type=str, required=False,default="/home/yueyulin/models/whisper-large-v3/")
@click.option("--audio_path", type=str, required=False,default="/home/yueyulin/github/RWKVTTS/my_chinese.wav")
def generate_single_sample(audio_lm_path, llm_path, language, device, whisper_path, audio_path):
    whisper_feature_extractor, whisper_encoder = load_whisper_feature_extractor_and_encoder(whisper_path)
    audio_lm_model = RWKV7ModelForLatentInputs.from_pretrained(audio_lm_path)
    print(f"Loaded audio_lm_model: {audio_lm_model}")
    llm = RWKV7ForCausalLM.from_pretrained(llm_path)
    print(f"Loaded llm: {llm}")
    asr_model = RWKV7ASRModel(whisper_encoder,audio_lm_model, llm, whisper_feature_extractor)
    asr_model.projector1.load_state_dict(torch.load(os.path.join(audio_lm_path, 'projector1.pt')),strict=True)
    asr_model.projector2.load_state_dict(torch.load(os.path.join(audio_lm_path, 'projector2.pt')),strict=True)
    print(f'loaded projector1 and projector2')
    asr_model.eval()
    asr_model = asr_model.to(torch.float16).to(device)
    print(asr_model)
    tokenizer = AutoTokenizer.from_pretrained(llm_path,trust_remote_code=True)
    print(tokenizer)
    
    if language == 'chinese':
        print(f'language: {language}')
        instruction = "User: 请将以下语音转写为中文。\n"
        hints = "Assistant: "
    else:
        print(f'language: {language}')
        instruction = "User: Convert the audios to English.\n"
        hints = "Assistant: "
    print(f'instruction: {instruction}')
    print(f'hints: {hints}')
    instruction_input_ids = tokenizer.encode(instruction)
    hints_input_ids = tokenizer.encode(hints)
    instruction_input_ids = torch.tensor(instruction_input_ids, dtype=torch.long,device=device).unsqueeze(0)
    hints_input_ids = torch.tensor(hints_input_ids, dtype=torch.long,device=device).unsqueeze(0)
    audio_path = audio_path
    audio_data,sample_rate = sf.read(audio_path)
    if sample_rate != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
    audio_data = audio_data.reshape(1,-1)
    print(f'type of audio_data: {type(audio_data)}')
    print(f'shape of audio_data: {audio_data.shape}')
    print(f'sample_rate: {sample_rate}')
    output = asr_model.inference_single(audio_data, instruction_input_ids, hints_input_ids)
    print(output)
    print(tokenizer.decode(output))
if __name__ == "__main__":
    generate_single_sample()
