import sparktts

from sparktts.models.audio_tokenizer import BiCodecTokenizer
model_dir = '/home/yueyulin/models/Spark-TTS-0.5B/'

audio_tokenizer = BiCodecTokenizer(model_dir)

print(audio_tokenizer)

audio_path = 'blade.wav'
from sparktts.utils.audio import load_audio
wav = load_audio(audio_path,sampling_rate=audio_tokenizer.config['sample_rate'],volume_normalize=audio_tokenizer.config["volume_normalize"],)
global_tokens, semantic_tokens = audio_tokenizer.tokenize(wav)

print(global_tokens)
print(semantic_tokens)

print(global_tokens.shape)
print(semantic_tokens.shape)

wav_reconstructed = audio_tokenizer.detokenize(global_tokens.squeeze(0), semantic_tokens)

reconstructed_audio_path = 'reconstructed_blade.wav'

import soundfile as sf

sf.write(reconstructed_audio_path, wav_reconstructed, audio_tokenizer.config['sample_rate'])
