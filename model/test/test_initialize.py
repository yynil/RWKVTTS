import time


def do_tts(tts_text,prompt_texts,prompt_audios,cosyvoice,prefix):
    import logging
    for i, (prompt_audio_file, prompt_text) in enumerate(zip(prompt_audios, prompt_texts)):
        logging.info(f'Processing {prompt_text}')
        prompt_speech_16k = load_wav(prompt_audio_file, 16000)
        if prompt_text is not None:
            for j, k in enumerate(cosyvoice.inference_zero_shot(tts_text,prompt_text, prompt_speech_16k, stream=False,speed=1)):
                torchaudio.save('{}_{}_{}.wav'.format(prefix,i, j), k['tts_speech'], cosyvoice.sample_rate)
        else:
            for j, k in enumerate(cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=False,speed=1)):
                torch.cuda.manual_seed_all(time.time())
                torchaudio.save('{}_{}_{}.wav'.format(prefix,i, j), k['tts_speech'], cosyvoice.sample_rate)
        logging.info(f'Finished processing {tts_text},for {i}th prompt')
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
        "/home/yueyulin/github/RWKVTTS/00000309-00000300.wav"
    ]
    
    if not is_flow_only:
        prompt_texts = [
            '希望你以后做的比我还好呦。',
            '少年强则中国强。',
            '我随便说一句话，我喊开始录就开始录。',
            'numbers of Latino, African American, Asian American and native American voters.',
            "小偷却一点也不气馁，继续在抽屉里翻找。"
        ]
    else:
        prompt_texts = [
            None,
            None,
            None,
            None
        ]
        
    tts_texts = [
        '一个教授逻辑学的教授，有三个学生，而且三个学生均非常聪明！一天教授给他们出了一个题，教授在每个人脑门上贴了一张纸条并告诉他们，每个人的纸条上都写了一个正整数，且某两个数的和等于第三个！',
        'By unifying streaming and non-streaming synthesis within a single framework, CosyVoice 2 achieves human parity naturalness, minimal response latency, and virtually lossless synthesis quality in streaming mode. '
        '全球每年有超过一百三十五万人，因交通事故而死亡。',
        '通过创新技术让未来出行更加安全，高效。'
    ]
    index = 0
    for tts_text in tts_texts:
        do_tts(tts_text,prompt_texts,prompt_audios,cosyvoice,f"MIXED{index}")
        index += 1
    
    # do_tts('By unifying streaming and non-streaming synthesis within a single framework, CosyVoice 2 achieves human parity naturalness, minimal response latency, and virtually lossless synthesis quality in streaming mode. ',prompt_texts,cosyvoice,"en")
    
    # do_tts('一个教授逻辑学的教授，有三个学生，而且三个学生均非常聪明！一天教授给他们出了一个题，教授在每个人脑门上贴了一张纸条并告诉他们，每个人的纸条上都写了一个正整数，且某两个数的和等于第三个！',prompt_texts,cosyvoice,"cn")