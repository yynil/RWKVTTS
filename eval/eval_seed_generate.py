#Download the evaluation file from:https://drive.google.com/file/d/1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP/edit
import os
voice_engine = None
def init_process_func(model_path,device):
    global voice_engine
    from cosyvoice.cli.cosyvoice import CosyVoice2  
    voice_engine = CosyVoice2(model_path,device=device,fp16=False,load_jit=False)
    print(f'Finish loading cosyvoice model from {model_path} in process {os.getpid()}')
def do_tts(ID,tts_text,prompt_text,prompt_audio_file,output_dir):
    from cosyvoice.utils.file_utils import load_wav
    import torchaudio
    global voice_engine
    try:
        final_output_file = os.path.join(output_dir,f'{ID}.wav')
        prompt_speech_16k = load_wav(prompt_audio_file, 16000)
        for output in voice_engine.inference_zero_shot(tts_text,prompt_text, prompt_speech_16k, stream=False,speed=1):
            torchaudio.save(final_output_file, output['tts_speech'], voice_engine.sample_rate)
            break # only save the first output
        print(f'TTS {tts_text} and Save to {final_output_file} at process {os.getpid()}')
    except Exception as e:
        print(f'Error: {e}')
        print(f'Error processing {ID} at process {os.getpid()}')
        import traceback
        traceback.print_exc()
        return
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default='eval_data/seedtts_testset')
    parser.add_argument("--language", type=str, default='zh',choices=['zh','en'])
    parser.add_argument("--model_path", type=str, default='/home/yueyulin/models/CosyVoice2-0.5B_RWKV_1.5B/')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--num_processes", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default='generated')
    parser.add_argument("--list_file", type=str, default='meta.lst')
    
    
    args = parser.parse_args()
    print(args)
    output_dir = os.path.join(args.eval_dir,args.language,args.output_dir)
    #first delete the output_dir
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    list_file = os.path.join(args.eval_dir,args.language,args.list_file)
    with open(list_file) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    print(f'Processing {len(lines)} lines')
    
    from multiprocessing import Pool
    from functools import partial
    import time
    with Pool(args.num_processes,init_process_func,(args.model_path,args.device)) as p:
        for line in lines:
            # 10002287-00000095|在此奉劝大家别乱打美白针。|prompt-wavs/10002287-00000094.wav|简单地说，这相当于惠普把消费领域市场拱手相让了。
            parts = line.split('|')
            ID = parts[0]
            tts_text = parts[3]
            prompt_text = parts[1]
            prompt_audio_file = os.path.join(args.eval_dir,args.language,parts[2])
            p.apply_async(do_tts,(ID,tts_text,prompt_text,prompt_audio_file,output_dir))
        p.close()
        p.join()
    print('All done')