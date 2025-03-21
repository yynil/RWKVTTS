import time


def do_tts(tts_text,ref_voice,prompt_texts,cosyvoice,prefix):
    import logging
    for i in range(len(prompt_texts)):
        prompt_text = prompt_texts[i]
        logging.info(f'Processing {prompt_text} from {ref_voice}')
        if len(prompt_text) >0:
            tts_text = f'{prompt_text}<|endofprompt|>{tts_text}'
        for result in cosyvoice.inference_sft(tts_text, ref_voice, stream=False,speed=1):
            torchaudio.save(f"{prefix}_{i}.wav", result['tts_speech'], cosyvoice.sample_rate)
        logging.info(f'Finished processing {tts_text} {prompt_text} from {ref_voice}')
if __name__ == '__main__':
    from cosyvoice.cli.cosyvoice import CosyVoice2
    import torch
    import sys
    # model_path = '/home/yueyulin/models/CosyVoice2-0.5B_RWKV_0.19B/'
    # device = 'cuda:0'
    print(sys.argv)
    model_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else 'cuda:0'
    cosyvoice = CosyVoice2(model_path,device=device,fp16=True,load_jit=False)
    print(cosyvoice.frontend.spk2info.keys())
    instruct = sys.argv[3]
    tts_text = sys.argv[4]
    ref_voice = sys.argv[5]
    from cosyvoice.utils.file_utils import load_wav
    import torchaudio
    
    prompt_texts = [
            # '希望你以后做的比我还好呦。',
            # '请用非常快速的语速说。',
            '日本語で話してください。',
            # '한국어로 말씀해주세요.'
        ]
    
    # do_tts('By unifying streaming and non-streaming synthesis within a single framework, CosyVoice 2 achieves human parity naturalness, minimal response latency, and virtually lossless synthesis quality in streaming mode. ',prompt_texts,cosyvoice,"instructed_en")
    
    # do_tts('[laughter]有时候，看着小孩子们的天真行为[laughter]，我们总会会心一笑。',prompt_audios,prompt_texts,cosyvoice,"instructed_cn")
    # do_tts(tts_text,prompt_audios,prompt_texts,cosyvoice,"instructed_cn")
    do_tts(tts_text,ref_voice,[instruct],cosyvoice,"speaker_info_test")
    