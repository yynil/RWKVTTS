import onnxruntime
import torch
import whisper
import numpy as np


def create_speech_tokenizer(model_path,device_id=0):
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    speech_tokenizer_session = onnxruntime.InferenceSession(model_path, sess_options=option,
                                                                     providers=[("CUDAExecutionProvider", {"device_id": device_id}) if torch.cuda.is_available() else
                                                                                "CPUExecutionProvider"])
    return speech_tokenizer_session

def extract_speech_token(speech_tokenizer_session,speech):
    # 确保音频长度不超过30秒
    assert speech.shape[1] / 16000 <= 30, 'do not support extract speech token for audio longer than 30s'
    # 转换为torch tensor并计算mel spectrogram
    speech = torch.from_numpy(speech)
    feat = whisper.log_mel_spectrogram(speech, n_mels=128)
    # 运行tokenizer
    speech_token = speech_tokenizer_session.run(None,
                                              {speech_tokenizer_session.get_inputs()[0].name:
                                               feat.detach().cpu().numpy(),
                                               speech_tokenizer_session.get_inputs()[1].name:
                                               np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
    speech_token = torch.tensor([speech_token], dtype=torch.int32)
    speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32)
    return speech_token, speech_token_len

def extract_text_token(text,tokenizer):
    text_token = tokenizer.encode(text)
    text_token = torch.tensor([text_token], dtype=torch.int32)
    text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32)
    return text_token, text_token_len

def collate_fn(batch_list, speech_tokenizer_session, text_tokenizer):
    # 初始化批次数据
    batch = {
        'text_token': [],
        'text_token_len': [],
        'speech_token': [],
        'speech_token_len': []
    }
    
    # 处理每个样本并记录最大长度
    max_text_len = 0
    max_speech_len = 0
    
    # 首先获取所有token并找到最大长度
    processed_items = []
    for item in batch_list:
        # 处理音频
        speech = item['audio']['array']
        speech = np.expand_dims(speech, axis=0).astype(np.float32)
        speech_token, speech_token_len = extract_speech_token(speech_tokenizer_session, speech)
        
        # 处理文本
        text = item['json']['text']
        text_token, text_token_len = extract_text_token(text, text_tokenizer)
        
        # 更新最大长度
        max_text_len = max(max_text_len, text_token.size(1))
        max_speech_len = max(max_speech_len, speech_token.size(1))
        
        processed_items.append({
            'speech_token': speech_token,
            'speech_token_len': speech_token_len,
            'text_token': text_token,
            'text_token_len': text_token_len
        })
    
    # 对所有样本进行padding
    for item in processed_items:
        # Pad text tokens
        text_pad = torch.zeros((1, max_text_len), dtype=torch.int32)
        text_pad[0, :item['text_token'].size(1)] = item['text_token'][0]
        batch['text_token'].append(text_pad)
        batch['text_token_len'].append(item['text_token_len'])
        
        # Pad speech tokens
        speech_pad = torch.zeros((1, max_speech_len), dtype=torch.int32)
        speech_pad[0, :item['speech_token'].size(1)] = item['speech_token'][0]
        batch['speech_token'].append(speech_pad)
        batch['speech_token_len'].append(item['speech_token_len'])
    
    # 将列表转换为张量
    batch['speech_token'] = torch.cat(batch['speech_token'], dim=0)
    batch['speech_token_len'] = torch.cat(batch['speech_token_len'], dim=0)
    batch['text_token'] = torch.cat(batch['text_token'], dim=0)
    batch['text_token_len'] = torch.cat(batch['text_token_len'], dim=0)
    
    return batch

if __name__ == '__main__':
    speech_target_sample_rate = 16000
    from data.spark.multiple_webdataset import MultipleWebDataset
    device_id = 0
    device = f'cuda:{device_id}'
    ds = MultipleWebDataset(data_dir='/home/yueyulin/data/voxbox_wids/',target_sr=speech_target_sample_rate,target_channels=1,shuffle=True,verify_tar=False)
    print(ds)
    speech_tokenizer_session = create_speech_tokenizer(model_path='/home/yueyulin/models/CosyVoice2-0.5B-RWKV-7-1.5B-Instruct-CHENJPKO-HF/speech_tokenizer_v2.onnx',device_id=device_id)
    print(speech_tokenizer_session)
    print(ds[0])
    speech_16k = ds[0]['audio']['array']
    speech_16k = np.expand_dims(speech_16k, axis=0).astype(np.float32)
    print(speech_16k.shape)
    speech_token, speech_token_len = extract_speech_token(speech_tokenizer_session,speech_16k)
    print(speech_token)
    print(speech_token_len)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/home/yueyulin/models/CosyVoice2-0.5B-RWKV-7-1.5B-Instruct-CHENJPKO-HF/ConvertedCosyLLM/',trust_remote_code=True)
    json_obj = ds[0]['json']
    text = json_obj['text']
    text_token, text_token_len = extract_text_token(text,tokenizer)
    print(text_token)
    print(text_token_len)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained('/home/yueyulin/models/CosyVoice2-0.5B-RWKV-7-1.5B-Instruct-CHENJPKO-HF/ConvertedCosyLLM/',trust_remote_code=True)
    dtype = torch.bfloat16
    model = model.to(dtype).to(device)
    print(model)
    model.train()

    from functools import partial
    from torch.utils.data import DataLoader
    batch_size = 64
    max_tokens_k = 4
    collate_fn = partial(collate_fn, speech_tokenizer_session=speech_tokenizer_session, text_tokenizer=tokenizer)
    dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=1e-4)
    for batch in dataloader:
        optimizer.zero_grad()
        print(batch)
        batch = {
            'text_token': batch['text_token'].to(device),
            'text_token_len': batch['text_token_len'].to(device),
            'speech_token': batch['speech_token'].to(device),
            'speech_token_len': batch['speech_token_len'].to(device)
        }
        output = model(batch=batch,use_cache=False,max_tokens_k=max_tokens_k)
        print(output)
        loss = output[0]
        loss.backward()
        optimizer.step()
        print(loss)
        break