from turtle import back
from click import prompt
import torch
from cosyvoice.cli.cosyvoice import CosyVoice2
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
model_path = '/data/yueyu/models/CosyVoice2-0.5B'
# cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=False)
# print(cosyvoice)
# from cosyvoice.utils.file_utils import load_wav
# import torchaudio
# prompt_speech_16k = load_wav('/home/yueyulin/github/CosyVoice/asset/zero_shot_prompt.wav', 16000)
# # prompt_speech_16k = torch.rand((1, 16000))
# for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    
# for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
#     torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
# # instruct usage
# for i, j in enumerate(cosyvoice.inference_instruct2('吾今朝早上去外婆家吃饭。', '用上海话说这句话', prompt_speech_16k, stream=False)):
#     torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

from hyperpyyaml import load_hyperpyyaml
import os
def load_from_configuration(model_dir):
    with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
    return configs

configs = load_from_configuration(model_path)
print(configs)

import torchaudio
def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav, backend='soundfile')
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech

zh_prompt_tar_file="/data/yueyu/data/Emilia-Dataset/Emilia/ZH/ZH-B000000.tar"
en_prompt_tar_file="/data/yueyu/data/Emilia-Dataset/Emilia/EN/EN-B000000.tar"


def load_file_list(tar_file):
    #the files are FILE_NAME.mp3/FILE_NAME.json
    #return all FILE_NAME as a list which has a mp3 and json
    import tarfile
    with tarfile.open(tar_file, 'r') as f:
        file_names = f.getnames()
    mp3_files = [i for i in file_names if i.endswith('.mp3')]
    json_files = [i for i in file_names if i.endswith('.json')]
    
    #filter mp3_files without corresponded json
    mp3_files = [i for i in mp3_files if i.replace('.mp3', '.json') in json_files] 
    return mp3_files

zh_files = load_file_list(zh_prompt_tar_file)
print(zh_files[:10])
en_files = load_file_list(en_prompt_tar_file)
print(en_files[:10])
import io

def load_random_samples_from_tar(tar_file, files, num_samples,target_sr,max_duration=10):
    import random
    import tarfile
    import json
    samples = []
    with tarfile.open(tar_file, 'r') as f:
        for i in random.sample(files, len(files)):
            mp3 = f.extractfile(i)
            mp3_bytes = io.BytesIO(mp3.read())
            speech, sample_rate = torchaudio.load(mp3_bytes,backend='soundfile')
            json_file = f.extractfile(i.replace('.mp3', '.json'))
            json_data = json.load(json_file)
            duration = json_data['duration']
            if duration > max_duration:
                continue
            speech = speech.mean(dim=0, keepdim=True)
            if sample_rate != target_sr:
                assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
                speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
            samples.append((speech, json_data,sample_rate))
            if len(samples) == num_samples:
                break
    return samples
target_sr = 16000
zh_samples = load_random_samples_from_tar(zh_prompt_tar_file, zh_files, 10, target_sr)

one_sample,one_json,sample_rate = zh_samples[0]
print(one_json)
print(sample_rate)
torchaudio.save('zh_sample.wav', one_sample, target_sr)
print(len(zh_samples))

en_samples = load_random_samples_from_tar(en_prompt_tar_file, en_files, 10, target_sr)
one_sample,one_json,sample_rate = en_samples[0]
print(one_json)
print(sample_rate)
torchaudio.save('en_sample.wav', one_sample, target_sr)
print(len(en_samples))

def resample_audio(samples, target_sr):
    resampled_samples = []
    for i in samples:
        speech, sample_rate = i
        if sample_rate != target_sr:
            assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
            speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
        resampled_samples.append((speech, sample_rate))
    return resampled_samples

prompt_text = zh_samples[0][1]['text']
prompt_speech = zh_samples[0][0]
print(prompt_text)
print(prompt_speech)
from cosyvoice.cli.cosyvoice import CosyVoice2
cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=True)
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
frontend = cosyvoice.frontend
prompt_text = frontend.text_normalize(prompt_text,split=False, text_frontend=True)
print(f'normalized prompt_text:{prompt_text}')
tts_text = '扫一扫，立即体验中国银行信用卡好礼、绑卡立减等热门活动，实时掌握更多优惠信息。'
tts_text = "在中国的一个偏远山区，有一位名叫李远的年轻人，他对集群通信系统有着浓厚的兴趣。每天晚上，他都会在自己的小屋里研究各种关于集群通信系统的资料，试图弄懂其中的原理和运作机制。他对这个领域的研究不仅仅停留在理论层面，还亲手制作了一些模型，试图通过实践来加深理解。"
tts_text = "歷史（现代汉语词汇，古典文言文称之为史），指人类社会过去的事件和行动，以及对这些事件行为有系统的记录、诠释和研究。歷史可提供今人理解過去，作為未來行事的參考依據，与伦理、哲学和艺术同属人类精神文明的重要成果。历史的第二个含义，即对过去事件的记录和研究，又称历史学”，或简称“史学”。隶属于历史学或与其密切相关的学科有年代学、编纂学、家谱学、古文字学、计量历史学、考古学、社会学和新闻学等，参见历史学。记录和研究历史的人称为历史学家，简称“史学家”，中国古代称为史官。记录历史的书籍称为史书，如《史記》、《汉书》等，粗分為「官修」與「民載」兩類。"
tts_text = frontend.text_normalize(tts_text,split=False, text_frontend=True)
print(f'normalized tts_text:{tts_text}')
final_rate = 24000
model_input = frontend.frontend_zero_shot(tts_text, prompt_text, prompt_speech,final_rate)
print(model_input)
llm = cosyvoice.model.llm
device = cosyvoice.model.device
text = model_input['text'].to(device)
text_len = torch.tensor([text.shape[1]], dtype=torch.int32).to(device)
prompt_text = model_input['prompt_text'].to(device)
prompt_text_len = torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(device)
llm_prompt_speech_token = model_input['llm_prompt_speech_token'].to(device)
prompt_speech_token_len = torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(device)
flow_prompt_speech_token = model_input['flow_prompt_speech_token'].to(device)
prompt_speech_feat = model_input['prompt_speech_feat'].to(device)
llm_embedding = model_input['llm_embedding'].to(device)
flow_embedding = model_input['flow_embedding'].to(device)
speech_tokens = []
for i in llm.inference(text = text, 
                       text_len = text_len, 
                       prompt_text = prompt_text,
                       prompt_text_len = prompt_text_len,
                       prompt_speech_token = llm_prompt_speech_token,
                       prompt_speech_token_len = prompt_speech_token_len,
                       embedding=llm_embedding
                       ):
    speech_tokens.append(i)
print(speech_tokens)

tts_speech_tokens = torch.tensor(speech_tokens).unsqueeze(dim=0).to(device)
print(f'tts_speech_tokens shape:{tts_speech_tokens.shape}')
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
torchaudio.save('zh_tts.wav', tts_speech, final_rate)