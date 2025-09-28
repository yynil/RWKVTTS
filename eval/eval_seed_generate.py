#Download the evaluation file from:https://drive.google.com/file/d/1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP/edit
import os

from eval_tts_base import create_tts_engine
import gc
import torch




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default='eval_data/seedtts_testset')
    parser.add_argument("--language", type=str, default='zh', choices=['zh', 'en'])
    parser.add_argument("--model_path", type=str, default='/home/yueyulin/models/rwkv7-0.4B-g1-respark-voice-tunable-25k/')
    parser.add_argument("--audio_tokenizer_path", type=str, default='/home/yueyulin/models/Spark-TTS-0.5B/')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--output_dir", type=str, default='eval_results')
    parser.add_argument("--list_file", type=str, default='meta.lst')
    args = parser.parse_args()
    print(args)
    output_dir = os.path.join(args.output_dir, args.language)
    
    # 首先删除输出目录
    if os.path.exists(output_dir):  
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    list_file = os.path.join(args.eval_dir, args.language, args.list_file)
    with open(list_file) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    print(f'Processing {len(lines)} lines')
    
    # 创建 Respark 引擎实例
    engine = create_tts_engine(
        engine_type="respark",
        device=args.device,
        model_path=args.model_path,
        audio_tokenizer_path=args.audio_tokenizer_path,
        language=args.language
    )
    for i, line in enumerate(lines):
        print(f"处理进度: {i+1}/{len(lines)}")
        
        # 10002287-00000095|在此奉劝大家别乱打美白针。|prompt-wavs/10002287-00000094.wav|简单地说，这相当于惠普把消费领域市场拱手相让了。
        parts = line.split('|')
        ID = parts[0]
        tts_text = parts[3]
        prompt_text = parts[1]
        prompt_audio_file = os.path.join(args.eval_dir, args.language, parts[2])
        final_output_file = os.path.join(output_dir, f'{ID}.wav')
        success = engine.do_tts(tts_text=tts_text, 
                                prompt_text=prompt_text, 
                                prompt_audio_file=prompt_audio_file, 
                                final_output_file=final_output_file)
        print(f"TTS 操作成功: {final_output_file} and {tts_text} and {prompt_text} and {prompt_audio_file}")