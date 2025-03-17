import time


def do_tts(tts_text,prompt_audios,prompt_texts,cosyvoice,prefix):
    import logging
    for i in range(len(prompt_audios)):
        prompt_audio_file = prompt_audios[i]
        prompt_speech_16k = load_wav(prompt_audio_file, 16000)
        for j in range(len(prompt_texts)):
            prompt_text = prompt_texts[j]
            logging.info(f'Processing {prompt_text} from {prompt_audio_file}')
            torch.cuda.manual_seed_all(time.time())
            for result in cosyvoice.inference_instruct2(tts_text,prompt_text, prompt_speech_16k, stream=False,speed=1):
                torchaudio.save(f"{prefix}_{i}_{j}.wav", result['tts_speech'], cosyvoice.sample_rate)
            logging.info(f'Finished processing {prompt_text} from {prompt_audio_file}')
if __name__ == '__main__':
    from cosyvoice.cli.cosyvoice import CosyVoice2
    import torch
    import sys
    # model_path = '/home/yueyulin/models/CosyVoice2-0.5B_RWKV_0.19B/'
    # device = 'cuda:0'
    print(sys.argv)
    model_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else 'cuda:0'
    cosyvoice = CosyVoice2(model_path,device=device,fp16=False,load_jit=False)
    
    from cosyvoice.utils.file_utils import load_wav
    import torchaudio
    prompt_audios = [
        # '/home/yueyulin/github/RWKVTTS/zero_shot_prompt.wav',
        '/home/yueyulin/github/RWKVTTS/mine.wav',
        # '/home/yueyulin/github/RWKVTTS/new.wav',
        # '/home/yueyulin/github/RWKVTTS/Trump.wav',
    ]
    
    prompt_texts = [
            '尝试一下以小猪佩奇的角色和我交流。',
            '请用非常快速的语速说。',
            '请用四川话和我说话。'
        ]
    
    # do_tts('By unifying streaming and non-streaming synthesis within a single framework, CosyVoice 2 achieves human parity naturalness, minimal response latency, and virtually lossless synthesis quality in streaming mode. ',prompt_texts,cosyvoice,"instructed_en")
    
    do_tts('一个教授逻辑学的教授，[laughter]有三个学生，而且三个学生均非常聪明！[breath]一天教授给他们出了一个题，[breath]教授在每个人<strong>脑门上</strong>贴了一张纸条并告诉他们，每个人的纸条上都写了一个正整数，且某两个数的和等于第三个！',prompt_audios,prompt_texts,cosyvoice,"instructed_cn")