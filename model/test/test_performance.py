from math import cos

from regex import F


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
    print('start to compile llm')
    import torch_tensorrt
    cosyvoice.model.llm = torch.compile(cosyvoice.model.llm,backend='tensorrt')
    print('finish to compile llm')
    tts_text = "\"Ra Kuv\" is an architecture that builds various kinds of neural network models."
    ref_voice = "Cocona_English"
    import torchaudio
    i = 0
    
    
    for result in cosyvoice.inference_sft(tts_text, ref_voice, stream=False,speed=1):
        torchaudio.save(f"performance_{i}.wav", result['tts_speech'], cosyvoice.sample_rate)
        i += 1
    cosyvoice.model.llm.start_to_profile()
    for i in range(100):
        print(f'Processing {tts_text} from {ref_voice} {i} times')
        for result in cosyvoice.inference_sft(tts_text, ref_voice, stream=False,speed=1):
            continue
    cosyvoice.model.llm.print_usage()
    cosyvoice.model.llm.finish_to_profile()