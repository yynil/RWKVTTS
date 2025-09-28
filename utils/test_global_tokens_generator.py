from utils.utilities import generate_global_tokens,generate_input_embeddings,extract_embeddings_for_global_tokens
import click
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnglishNormalizer
chinese_normalizer = ZhNormalizer(remove_erhua=False, full_to_half=False, overwrite_cache=False, remove_interjections=False)
english_normalizer = EnglishNormalizer()

@click.command()
@click.option("--model_path", type=str, required=True)
@click.option("--ckpt_file", type=str, required=False,default=None)
@click.option("--text", type=str, required=False,default="这是一个悲伤的故事。白毛女从小家境贫寒，父母双亡，被地主黄世仁家收养。")
@click.option("--age", type=str, required=False,default="Elderly")
@click.option("--gender", type=str, required=False,default="male")
@click.option("--emotion", type=str, required=False,default="ANGRY")
@click.option("--pitch", type=str, required=False,default="medium_pitch")
@click.option("--speed", type=str, required=False,default="medium")
@click.option("--device", type=str, required=False,default="cuda:0")
@click.option("--trained_model_path", type=str, required=False,default=None)
@click.option("--audio_tokenizer_path", type=str, required=False,default="/home/yueyulin/models/Spark-TTS-0.5B/")
@click.option("--use_properties_to_generate_semantic_tokens", type=bool, required=False,default=False)
def main(model_path,ckpt_file,text,age,gender,emotion,pitch,speed,device,trained_model_path,audio_tokenizer_path,use_properties_to_generate_semantic_tokens):
    text = chinese_normalizer.normalize(text) 
    print(f'normalized text: {text}')
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if ckpt_file is not None:
        model.load_state_dict(torch.load(ckpt_file,map_location="cpu"))
    model = model.to(dtype)
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = generate_global_tokens(model, tokenizer, text, age, gender, emotion, pitch, speed)
    print(outputs)
    global_tokens = outputs['sequences']
    global_tokens = global_tokens.squeeze(0).tolist()
    print(f'global_tokens: {global_tokens}')
    if trained_model_path is None:
        traind_model = model
    else:
        traind_model = AutoModelForCausalLM.from_pretrained(trained_model_path, trust_remote_code=True)
        traind_model = traind_model.to(dtype)
        traind_model.to(device)
        traind_model.eval() 
    with torch.no_grad():
        if trained_model_path is not None:
            input_embs = generate_input_embeddings(traind_model, tokenizer, text, global_tokens)
            cache = None
        else:
            if use_properties_to_generate_semantic_tokens:
                input_embs = extract_embeddings_for_global_tokens(traind_model, tokenizer, text, age, gender, emotion, pitch, speed,global_tokens=global_tokens)
            else:
                input_embs = generate_input_embeddings(traind_model, tokenizer, text, global_tokens)
            cache = None
    input_embs = input_embs.unsqueeze(0)
    print(input_embs)
    print('input_embs shape: ',input_embs.shape)
    gen_args = {
        "inputs_embeds":input_embs,
        "attention_mask":torch.ones((1, input_embs.shape[1]),dtype=torch.long,device=device),
        "max_new_tokens":1024,
        "do_sample":True,
        "top_k":50,
        "top_p":0.95,
        "temperature":1.0,
        "eos_token_id":model.config.vocab_size-1,
        "pad_token_id":tokenizer.pad_token_id,
        "use_cache":True,
    }
    with torch.no_grad():
        generated_outputs = traind_model.generate(**gen_args)
    print(generated_outputs)
    global_tokens = torch.tensor(global_tokens,dtype=torch.long,device=device).unsqueeze(0)
    semantic_tokens = generated_outputs[:,:-1]
    print(f'global shape: {global_tokens.shape}, semantic shape: {semantic_tokens.shape}')
    from sparktts.models.audio_tokenizer import BiCodecTokenizer
    audio_tokenizer = BiCodecTokenizer(audio_tokenizer_path, device)
    wav_reconstructed = audio_tokenizer.detokenize(global_tokens, semantic_tokens)

    import soundfile as sf
    sf.write("from_generated_global_tokens.wav", wav_reconstructed, audio_tokenizer.config['sample_rate'])
if __name__ == "__main__":
    main()
