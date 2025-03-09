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
    model_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else 'cuda:0'
    cosyvoice = CosyVoice2(model_path,device=device,fp16=False,load_jit=False)
    
    from cosyvoice.utils.file_utils import load_wav
    import torchaudio
    prompt_audios = [
        '/home/yueyulin/github/RWKVTTS/zero_shot_prompt.wav',
        '/home/yueyulin/github/RWKVTTS/mine.wav',
        '/home/yueyulin/github/RWKVTTS/new.wav',
        '/home/yueyulin/github/RWKVTTS/Trump.wav',
    ]
    prompt_texts = [
        '希望你以后做的比我还好呦。',
        '少年强则中国强。',
        '我随便说一句话，我喊开始录就开始录。',
        'numbers of Latino, African American, Asian American and native American voters.'
    ]
    prompt_texts = [
        None,
        None,
        None,
        None
    ]
    do_tts('Make America great again!',prompt_texts,cosyvoice)