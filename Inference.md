# Install the code base and the dependencies
```bash
git clone https://github.com/yynil/RWKVTTS
git clone https://github.com/yynil/CosyVoice
```
Add these two directories to the PYTHONPATH
```bash
export PYTHONPATH=$PYTHONPATH:/home/user/CosyVoice:/home/user/RWKVTTS
```
# Install the dependencies
```bash
conda create -n rwkvtts-311 -y python=3.11
conda activate rwkvtts-311
conda install -y -c conda-forge pynini==2.1.6
cd RWKVTTS
pip install -r rwkvtts_requirements.txt
``` 

Download the pretrained models from the following links:
https://huggingface.co/yueyulin/rwkv-tts-base

Place the CosyVoice2-0.5B_RWKV_0.19B to local directory. Let's say /home/user/CosyVoice2-0.5B_RWKV_0.19B

Add two directories to the PYTHONPATH

The example code for inference is as follows:
```python
if __name__ == '__main__':
    from cosyvoice.cli.cosyvoice import CosyVoice2
    import torch
    model_path = '/home/yueyulin/models/CosyVoice2-0.5B_RWKV_0.19B/'
    device = 'cuda:0'
    cosyvoice = CosyVoice2(model_path,device=device,fp16=True,load_jit=True)
    
    from cosyvoice.utils.file_utils import load_wav
    import torchaudio
    prompt_audio_file = '/home/yueyulin/github/RWKVTTS/zero_shot_prompt.wav'
    prompt_audio_file = '/home/yueyulin/github/RWKVTTS/mine.wav'
    prompt_audio_file = '/home/yueyulin/github/RWKVTTS/another.wav'
    prompt_speech_16k = load_wav(prompt_audio_file, 16000)
    prompt_text = '希望你以后做的比我还好呦。'
    prompt_text = '今天天气挺不错的。'
    prompt_text = '我家里有三只狗。'
    cosyvoice.model.llm.dummy_forward()
    print('Finished warmup')
    prompt_audios = [
        '/home/yueyulin/github/RWKVTTS/zero_shot_prompt.wav',
        '/home/yueyulin/github/RWKVTTS/mine.wav',
        '/home/yueyulin/github/RWKVTTS/new.wav'
    ]
    prompt_texts = [
        '希望你以后做的比我还好呦。',
        '少年强则中国强。',
        '我随便说一句话，我喊开始录就开始录。'
    ]
    import logging
    tts_text = '中国在东亚，是世界上最大的国家，也是世界上人口最多的国家。'
    for i, (prompt_audio_file, prompt_text) in enumerate(zip(prompt_audios, prompt_texts)):
        logging.info(f'Processing {prompt_text}')
        prompt_speech_16k = load_wav(prompt_audio_file, 16000)
        with torch.no_grad():
            for j, k in enumerate(cosyvoice.inference_zero_shot(tts_text,prompt_text, prompt_speech_16k, stream=False,speed=1.)):
                torchaudio.save('zero_{}_{}.wav'.format(i, j), k['tts_speech'], cosyvoice.sample_rate)
        logging.info(f'Finished processing {prompt_text}')
```
Please change the paths to the correct paths in your system.

You can also use your own prompt audio and text. Since the llm module is to finish your audio tokens for you, so please make sure the audio is clean,complete and the text is correct. Otherwise, the result may not be good.

The following table shows the example results of the above code:
| Prompt Audio | Prompt Text | TTS Text | Result |
| --- | --- | --- | --- |
| zero_shot_prompt.wav | 希望你以后做的比我还好呦。 | 中国在东亚，是世界上最大的国家，也是世界上人口最多的国家。 | zero_0_0.wav |
| mine.wav | 少年强则中国强。 | 中国在东亚，是世界上最大的国家，也是世界上人口最多的国家。 | zero_1_0.wav |
| new.wav | 我随便说一句话，我喊开始录就开始录。 | 中国在东亚，是世界上最大的国家，也是世界上人口最多的国家。 | zero_2_0.wav |
