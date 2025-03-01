from pyexpat import model
import torchaudio
from hyperpyyaml import load_hyperpyyaml
import os
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.cosyvoice import CosyVoice2
import json
import torch

def load_from_configuration(model_dir):
    with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
    return configs
def init_process(model_dir,device):
    cosyvoice = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=True,device=device)
    # configs = load_from_configuration(model_dir)
    # frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
    #                                       configs['feat_extractor'],
    #                                       '{}/campplus.onnx'.format(model_dir),
    #                                       '{}/speech_tokenizer_v2.onnx'.format(model_dir),
    #                                       '{}/spk2info.pt'.format(model_dir),
    #                                       configs['allowed_special'],
    #                                       device)
    frontend = cosyvoice.frontend
    llm = cosyvoice.model.llm
    return frontend,llm,cosyvoice


def preprocess_prompts(frontend,prompts_dir):
    language_results = {}
    #iterator all json file in prompts_dir, extract the json contents, get the language from the json. load the wav file and preprocess it
    #put the (text,wav) into language_results
    final_rate = 24000
    for root, dirs, files in os.walk(prompts_dir):
        for file in files:
            if file.endswith('.json'):
                json_file = os.path.join(root, file)
                language = json_file.split('/')[-2]
                if language not in language_results:
                    language_results[language] = []
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                wav_file = json_file.replace('.json', '.wav')
                prompt_text = json_data['text']
                prompt_speech = torchaudio.load(wav_file, backend='soundfile')[0]
                fake_tts_text = "a"
                with torch.no_grad():
                    model_input = frontend.frontend_zero_shot(fake_tts_text, prompt_text, prompt_speech,final_rate)
                language_results[language].append((model_input,prompt_text))
    return language_results

def generate_speech_tokens(llm,frontend,tts_text,model_input,device):
    tts_text = frontend.text_normalize(tts_text,split=False, text_frontend=True)
    tts_text_token, tts_text_token_len = frontend._extract_text_token(tts_text)
    tts_text_token_len = torch.tensor([tts_text_token.shape[1]], dtype=torch.int32).to(device)
    prompt_text = model_input['prompt_text'].to(device)
    prompt_text_len = torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(device)
    llm_prompt_speech_token = model_input['llm_prompt_speech_token'].to(device)
    prompt_speech_token_len = torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(device)
    flow_prompt_speech_token = model_input['flow_prompt_speech_token'].to(device)
    prompt_speech_feat = model_input['prompt_speech_feat'].to(device)
    llm_embedding = model_input['llm_embedding'].to(device)
    flow_embedding = model_input['flow_embedding'].to(device)
    speech_tokens = []
    for i in llm.inference(text = tts_text_token, 
                        text_len = tts_text_token_len, 
                        prompt_text = prompt_text,
                        prompt_text_len = prompt_text_len,
                        prompt_speech_token = llm_prompt_speech_token,
                        prompt_speech_token_len = prompt_speech_token_len,
                        embedding=llm_embedding
                        ):
        speech_tokens.append(i)
    tts_speech_tokens = torch.tensor(speech_tokens).unsqueeze(dim=0).to(device)
    return tts_speech_tokens

if __name__ == '__main__':
    model_dir = '/data/yueyu/models/CosyVoice2-0.5B'
    prompts_dir = 'extract_data/prompts'
    
    device = 'cuda:0'
    frontend,llm,cosyvoice = init_process(model_dir
                            ,device)
    prompts = preprocess_prompts(frontend,prompts_dir)
    print(prompts)
    model_input = prompts['zh'][0][0]
    prompt_text = prompts['zh'][0][1]
    tts_text = '扫一扫，立即体验中国银行信用卡好礼、绑卡立减等热门活动，实时掌握更多优惠信息。'
    tts_text = '在中国的一个偏远山区，有一位名叫李远的年轻人，他对集群通信系统有着浓厚的兴趣。每天晚上，他都会在自己的小屋里研究各种关于集群通信系统的资料，试图弄懂其中的原理和运作机制。他对这个领域的研究不仅仅停留在理论层面，还亲手制作了一些模型，试图通过实践来加深理解。'
    tts_text = "歷史（现代汉语词汇，古典文言文称之为史），指人类社会过去的事件和行动，以及对这些事件行为有系统的记录、诠释和研究。歷史可提供今人理解過去，作為未來行事的參考依據，与伦理、哲学和艺术同属人类精神文明的重要成果。历史的第二个含义，即对过去事件的记录和研究，又称历史学”，或简称“史学”。隶属于历史学或与其密切相关的学科有年代学、编纂学、家谱学、古文字学、计量历史学、考古学、社会学和新闻学等，参见历史学。记录和研究历史的人称为历史学家，简称“史学家”，中国古代称为史官。记录历史的书籍称为史书，如《史記》、《汉书》等，粗分為「官修」與「民載」兩類。"
    tts_text = "### 如何提高花样游泳水平"
    tts_speech_tokens = generate_speech_tokens(llm,frontend,tts_text,model_input,device)
    print(tts_speech_tokens)
    
    
    flow_prompt_speech_token = model_input['flow_prompt_speech_token'].to(device)
    prompt_speech_feat = model_input['prompt_speech_feat'].to(device)
    llm_embedding = model_input['llm_embedding'].to(device)
    flow_embedding = model_input['flow_embedding'].to(device)
    cosyvoice.model.hift_cache_dict['xxxx'] = None
    tts_speech = cosyvoice.model.token2wav(token=tts_speech_tokens,
                                                prompt_token=flow_prompt_speech_token,
                                                prompt_feat=prompt_speech_feat,
                                                embedding=flow_embedding,
                                                uuid='xxxx',
                                                token_offset=0,
                                                finalize=True,
                                                speed=1.0)
    print(f'tts_speech shape:{tts_speech.shape}')
    tts_speech = tts_speech.cpu()
    torchaudio.save('zh_tts_S.wav', tts_speech, 24000)
    print(model_input)