from math import cos

from regex import F


if __name__ == '__main__':
    from cosyvoice.cli.cosyvoice import CosyVoice2
    import torch
    import sys
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # model_path = '/home/yueyulin/models/CosyVoice2-0.5B_RWKV_0.19B/'
    # device = 'cuda:0'
    print(sys.argv)
    model_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else 'cuda:0'
    cosyvoice = CosyVoice2(model_path,device=device,fp16=True,load_jit=False)
    print('start to compile llm')
    import torch_tensorrt
    # cosyvoice.model.llm = torch.compile(cosyvoice.model.llm,backend='onnxrt')
    print('finish to compile llm')
    tts_text = "\"Ra Kuv\" is an architecture that builds various kinds of neural network models."
    ref_voice = "Cocona_English"
    import torchaudio
    i = 0
    
    # 重置 RWKV7Block 的统计信息
    from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Block
    RWKV7Block.reset_stats()
    
    # 重置 LoRA 的统计信息
    from rwkvfla.layers.rwkv6 import LoRA
    LoRA.reset_stats()
    
    # 重置 RWKV7Attention 的统计信息
    from rwkvfla.layers.rwkv7 import RWKV7Attention
    RWKV7Attention.reset_stats()
    for result in cosyvoice.inference_sft(tts_text, ref_voice, stream=False,speed=1):
        torchaudio.save(f"performance_{i}.wav", result['tts_speech'], cosyvoice.sample_rate)
        i += 1
    for i in range(100):
        print(f'Processing {tts_text} from {ref_voice} {i} times')
        for result in cosyvoice.inference_sft(tts_text, ref_voice, stream=False,speed=1):
            continue
    sort_by_all_time = True
    # 打印RWKV7Block内部统计信息
    from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Block
    RWKV7Block.print_stats(sort_by_time=sort_by_all_time)
    
    # 打印LoRA内部统计信息
    from rwkvfla.layers.rwkv6 import LoRA
    LoRA.print_stats()
    
    # 打印RWKV7Attention内部统计信息
    from rwkvfla.layers.rwkv7 import RWKV7Attention
    RWKV7Attention.print_stats(sort_by_time=sort_by_all_time)