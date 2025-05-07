import time


def do_tts(tts_text,prompt_audio,instruct,prompt_text,cosyvoice,prefix):
    import logging
    prompt_speech_16k = load_wav(prompt_audio, 16000)
    logging.info(f'Processing {instruct} from {prompt_audio}')
    torch.cuda.manual_seed_all(time.time())
    wav_cnt = 0
    if len(instruct) >0:
        print(f'instruct: {instruct}')
        for result in cosyvoice.inference_instruct2(tts_text,instruct, prompt_speech_16k, stream=False,speed=1,prompt_text = prompt_text):
            torchaudio.save(f"{prefix}_{wav_cnt}.wav", result['tts_speech'], cosyvoice.sample_rate)
            wav_cnt += 1
    else:
        print(f'tts_text: {tts_text}')
        for result in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False,speed=1):
            torchaudio.save(f"{prefix}_{wav_cnt}.wav", result['tts_speech'], cosyvoice.sample_rate)
            wav_cnt += 1
    logging.info(f'Finished processing {prompt_text} from {prompt_audio}')
'''
 520  python model/test/test_instructed.py /external_data/models/CosyVoice2-0.5B-RWKV-7-1.5B-Instruct-CHENJPKO/ cuda:0 "日本語で話してください。" "友達のあやかさんです。" /external_data/yueyudata/starrail-voice-top-japanese/Japanese_Acheron_4.wav "だが、あの時、私が刀を抜くことを選んでいたら"
  522  python model/test/test_instructed.py /external_data/models/CosyVoice2-0.5B-RWKV-7-1.5B-Instruct-CHENJPKO/ cuda:0 "한국어로 말씀해주세요." "좋습니다 좋아요" /external_data/yueyudata/starrail-voice-top-korean/Korean_Acheron_8.wav "긴장하지마,정상적인현상이야"
  523  python model/test/test_instructed.py /external_data/models/CosyVoice2-0.5B-RWKV-7-1.5B-Instruct-CHENJPKO/ cuda:0 "冷静地说." "I can do nothing but save all of you guys." /external_data/yueyudata/starrail-voice-top-english/English_Acheron_6.wav "This is the only way I can ensure everyone's safety."
  python model/test/test_instructed.py /external_data/models/CosyVoice2-0.5B-RWKV-7-1.5B-Instruct-CHENJPKO/ cuda:0 "你能模仿四川话的口音说吗?" "我们只有团结起来才能战胜所有敌人,冲出重围!" /external_data/yueyudata/genshin-voice-top/Chinese\(PRC\)_Acheron_3.wav "我们到了."
'''
if __name__ == '__main__':
    from cosyvoice.cli.cosyvoice import CosyVoice2
    import torch
    import sys
    # model_path = '/home/yueyulin/models/CosyVoice2-0.5B_RWKV_0.19B/'
    # device = 'cuda:0'
    print(sys.argv)
    model_path = sys.argv[1]
    device = sys.argv[2] 
    cosyvoice = CosyVoice2(model_path,device=device,fp16=True,load_jit=False)
    instruct = sys.argv[3]
    tts_text = sys.argv[4]
    prompt_audio = sys.argv[5] 
    prompt_text = sys.argv[6] 
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
    import os
    if os.path.exists(tts_text):
        print(f'read tts_text from {tts_text}')
        tts_text = open(tts_text).read()
        print(f'tts_text: {tts_text}')
    do_tts(tts_text,prompt_audio,instruct,prompt_text,cosyvoice,"instructed")
    