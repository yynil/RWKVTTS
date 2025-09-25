import click
import torch
from model.llm.rwkv_asr_whisper import RWKV7ASRModel,RWKV7ModelForLatentInputs
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Config, RWKV7Model, RWKV7ForCausalLM
from tokenizer.rwkv_tokenizer import RWKV_TOKENIZER
import transformers
from transformers import AutoTokenizer,WhisperFeatureExtractor,WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder
import soundfile as sf
import os
def load_whisper_feature_extractor_and_encoder(whisper_path):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_path)
    print(f"Loaded WhisperFeatureExtractor: {feature_extractor}")
    whisper_config = WhisperConfig.from_pretrained(whisper_path)
    print(f"Loaded WhisperConfig: {whisper_config}")
    encoder = WhisperEncoder(whisper_config)
    print(f"Created WhisperEncoder: {encoder}")
    return feature_extractor, encoder
@click.command()
@click.option("--audio_lm_path", type=str, required=True)
@click.option("--llm_path", type=str, required=True)
@click.option("--ckpt_path", type=str, required=True)
@click.option("--whisper_path", type=str, required=False,default="/home/yueyulin/models/whisper-large-v3/")
@click.option("--output_path", type=str, required=True)
def export_rwkv_asr_audio_lm(audio_lm_path, llm_path, ckpt_path, whisper_path, output_path):
    whisper_feature_extractor, whisper_encoder = load_whisper_feature_extractor_and_encoder(whisper_path)
    audio_lm_model_config = RWKV7Config.from_pretrained(audio_lm_path)
    print(f"Loaded audio_lm_model_config: {audio_lm_model_config}")
    with transformers.modeling_utils.no_init_weights():
        audio_lm_model = RWKV7ModelForLatentInputs(audio_lm_model_config)
    print(f"Loaded audio_lm_model: {audio_lm_model}")
    llm_config = RWKV7Config.from_pretrained(llm_path)
    print(f"Loaded llm_config: {llm_config}")
    with transformers.modeling_utils.no_init_weights():
        llm = RWKV7ForCausalLM(llm_config)
    print(f"Loaded llm: {llm}")
    asr_model = RWKV7ASRModel(whisper_encoder,audio_lm_model, llm, whisper_feature_extractor)
    info = asr_model.load_state_dict(torch.load(ckpt_path),strict=False)
    print(f'load ckpt info: {info}')
    asr_model.audio_lm_model.save_pretrained(output_path)
    print(f'saved audio_lm_model to {output_path}')
    print(f'done')
    torch.save(asr_model.projector1.state_dict(), os.path.join(output_path, 'projector1.pt'))
    torch.save(asr_model.projector2.state_dict(), os.path.join(output_path, 'projector2.pt'))
    print(f'saved projector1 and projector2 to {output_path}')
    print(f'done')
if __name__ == "__main__":
    export_rwkv_asr_audio_lm()