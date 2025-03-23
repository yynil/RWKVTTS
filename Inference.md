# Install the code base and the dependencies
```bash
git clone https://github.com/yynil/RWKVTTS
```
Add these two directories to the PYTHONPATH
```bash
export PYTHONPATH=$PYTHONPATH:/home/user/RWKVTTS:/home/user/RWKVTTS/third_party
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
https://huggingface.co/yueyulin/CosyVoice2-0.5B-RWKV-7-1.5B-Instruct-CHENJPKO

Place the CosyVoice2-0.5B-RWKV-7-1.5B-Instruct-CHENJPKO to local directory. Let's say /home/user/CosyVoice2-0.5B-RWKV-7-1.5B-Instruct-CHENJPKO

Add two directories to the PYTHONPATH

The example code for inference is as follows:
```python
def do_tts(tts_text,prompt_texts,cosyvoice):
    import logging
    for i, (prompt_audio_file, prompt_text) in enumerate(zip(prompt_audios, prompt_texts)):
        logging.info(f'Processing {prompt_text}')
        prompt_speech_16k = load_wav(prompt_audio_file, 16000)
        with torch.no_grad():
            if prompt_text is not None:
                for j, k in enumerate(cosyvoice.inference_zero_shot(tts_text,prompt_text, prompt_speech_16k, stream=False,speed=1)):
                    torchaudio.save('zero_{}_{}.wav'.format(i, j), k['tts_speech'], cosyvoice.sample_rate)
            else:
                for j, k in enumerate(cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=False,speed=1)):
                    torchaudio.save('zero_{}_{}.wav'.format(i, j), k['tts_speech'], cosyvoice.sample_rate)
        logging.info(f'Finished processing {prompt_text}')
if __name__ == '__main__':
    from cosyvoice.cli.cosyvoice import CosyVoice2
    import torch
    import sys
    # model_path = '/home/yueyulin/models/CosyVoice2-0.5B_RWKV_0.19B/'
    # device = 'cuda:0'
    print(sys.argv)
    model_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else 'cuda:0'
    is_flow_only = sys.argv[3]=='True' if len(sys.argv) > 3 else False
    print(f'is_flow_only: {is_flow_only}')
    cosyvoice = CosyVoice2(model_path,device=device,fp16=False,load_jit=False)
    
    from cosyvoice.utils.file_utils import load_wav
    import torchaudio
    prompt_audios = [
        '/home/yueyulin/github/RWKVTTS/zero_shot_prompt.wav',
        '/home/yueyulin/github/RWKVTTS/mine.wav',
        '/home/yueyulin/github/RWKVTTS/new.wav',
        '/home/yueyulin/github/RWKVTTS/Trump.wav',
    ]
    
    if not is_flow_only:
        prompt_texts = [
            '希望你以后做的比我还好呦。',
            '少年强则中国强。',
            '我随便说一句话，我喊开始录就开始录。',
            'numbers of Latino, African American, Asian American and native American voters.'
        ]
    else:
        prompt_texts = [
            None,
            None,
            None,
            None
        ]
    do_tts('Make America great again!',prompt_texts,cosyvoice)
```
More examples can be found in the model/test directory.
model/test/test_instructed.py is an example to use the instructed voice flow to generate the audio.
model/test/test_speaker_adapter.py is an example to use the speaker adapter to generate the audio.

If you pass the prompt_texts as None, the engine will only clone the voice flow and texture which is good to clone voice cross lingual. If you pass the correct prompt texts to the engine, the engine will try to continue to finish the audio tokens following the prompt audio you provided. This will be good to continue the audio you provided but it will be weird when you try to mix languages. 

The test source code is [test code](model/test/test_initialize.py).

Please change the paths to the correct paths in your system.

You can also use your own prompt audio and text. Since the llm module is to finish your audio tokens for you, so please make sure the audio is clean,complete and the text is correct. Otherwise, the result may not be good.

The following table shows the example results of the above code:
| Prompt Audio | Prompt Text | TTS Text | Result |
| --- | --- | --- | --- |
| https://github.com/yynil/RWKVTTS/raw/main/zero_shot_prompt.wav | 希望你以后做的比我还好呦。 | 中国在东亚，是世界上最大的国家，也是世界上人口最多的国家。 | https://github.com/yynil/RWKVTTS/raw/main/zero_0_0.wav |
| https://github.com/yynil/RWKVTTS/raw/main/mine.wav| 少年强则中国强。 | 中国在东亚，是世界上最大的国家，也是世界上人口最多的国家。 | https://github.com/yynil/RWKVTTS/raw/main/zero_1_0.wav |
| https://github.com/yynil/RWKVTTS/raw/main/new.wav | 我随便说一句话，我喊开始录就开始录。 | 中国在东亚，是世界上最大的国家，也是世界上人口最多的国家。 | https://github.com/yynil/RWKVTTS/raw/main/zero_2_0.wav |
