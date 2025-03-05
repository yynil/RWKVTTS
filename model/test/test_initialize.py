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