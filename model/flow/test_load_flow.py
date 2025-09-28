import os
from hyperpyyaml import load_hyperpyyaml
import yaml
import onnxruntime

from cosyvoice.utils.mask import make_pad_mask
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the YAML file
yaml_file = os.path.join(current_dir, 'test_flow.yaml')

with open(yaml_file, 'r') as f:
    configs = load_hyperpyyaml(f)

print(configs)
print(configs['flow'])

sample_rate = configs['sample_rate']
print(sample_rate)
feat_extractor = configs['feat_extractor']
print(feat_extractor)

#print feature_extractor function's signature
print(feat_extractor.__class__.__name__)
print(feat_extractor.__class__.__module__)
print(feat_extractor.__class__.__bases__)
print(feat_extractor.__class__.__dict__)
print(feat_extractor.__class__.__doc__)
print(feat_extractor.__class__.__module__)


data_dir = '/home/yueyulin/data/voxbox_wids/librispeech/'
import glob

data_files = glob.glob(os.path.join(data_dir, '*.tar'))
print(data_files)

from data.spark.multiple_webdataset import MultipleWebDataset
from matcha.utils.audio import mel_spectrogram
dataset = MultipleWebDataset(
    target_sr=sample_rate,
    data_files=data_files,
)
import torch
device_id = 0
device = f'cuda:{device_id}'
data0 = dataset[0]
print(data0)

speech = data0['audio']['array']
print(speech.shape)

speech = torch.from_numpy(speech).unsqueeze(0).to(torch.float).to(device)
print(speech.shape)

mel = feat_extractor(speech)
print(mel.shape)

import torchaudio

if sample_rate != 16000:
    speech_16k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(speech.cpu())
else:
    speech_16k = speech

campplus_path = '/home/yueyulin/models/CosyVoice2-0.5B/campplus.onnx'
providers=[("CUDAExecutionProvider", {"device_id": device_id})]
option = onnxruntime.SessionOptions()
option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
option.intra_op_num_threads = 1
campplus_session = onnxruntime.InferenceSession(campplus_path, sess_options=option, providers=providers)

import torchaudio.compliance.kaldi as kaldi
feat = kaldi.fbank(speech_16k,num_mel_bins=80,dither=0,sample_frequency=160000)
feat = feat - feat.mean(dim=0,keepdim=True)
feat = feat.unsqueeze(0)
print(feat.shape)
print(feat.dtype)
print(feat.device)
print(campplus_session.get_inputs()[0].name)
print(campplus_session.get_inputs()[0])
embedding = campplus_session.run(None, {campplus_session.get_inputs()[0].name: feat.numpy()})[0].flatten().tolist()
print(embedding)
speech_tokenizer_file = '/home/yueyulin/models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx'
speech_tokenizer_session = onnxruntime.InferenceSession(speech_tokenizer_file, sess_options=option, providers=providers)
print(data0)
print(speech_tokenizer_session.get_inputs()[0].name)
print(speech_tokenizer_session.get_inputs()[0])
print(speech_tokenizer_session.get_inputs()[1].name)
print(speech_tokenizer_session.get_inputs()[1])
import whisper
feat = whisper.log_mel_spectrogram(speech_16k,n_mels=128)
print(feat.shape)
print(feat.dtype)
print(feat.device)
import numpy as np
speech_token = speech_tokenizer_session.run(None,
                                                         {speech_tokenizer_session.get_inputs()[0].name:
                                                          feat.detach().cpu().numpy(),
                                                          speech_tokenizer_session.get_inputs()[1].name:
                                                          np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
print(speech_token)

# 定义自定义的collate函数来处理不同长度的音频数据
def simple_collate_fn(batch):
    """
    简单的collate函数，返回原始数据列表而不是尝试堆叠
    """
    return batch


def process_one_batch(batch,feat_extractor,campplus_session,speech_tokenizer_session,sample_rate):
    mel_list = []
    embedding_list = []
    speech_token_list = []
    texts_list = []
    for i, data in enumerate(batch):
        print(f"\n--- Processing sample {i+1} ---")
        speech = data['audio']['array']
        speech = torch.from_numpy(speech).unsqueeze(0).to(torch.float).to(device)
        mel = feat_extractor(speech).squeeze(0)
        if sample_rate != 16000:
            speech_16k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(speech.cpu())
        else:
            speech_16k = speech
        
        # 处理campplus embedding
        feat = kaldi.fbank(speech_16k,num_mel_bins=80,dither=0,sample_frequency=160000)
        feat = feat - feat.mean(dim=0,keepdim=True)
        feat = feat.unsqueeze(0)
        embedding = campplus_session.run(None, {campplus_session.get_inputs()[0].name: feat.numpy()})[0].flatten().tolist()
        print(f'embedding shape: {len(embedding)}')
        embedding_list.append(torch.tensor(embedding,dtype=torch.bfloat16))

        # 处理speech tokenizer - 使用正确的whisper mel spectrogram
        print(f"Original speech shape: {speech_16k.shape}")
        print(f"Speech range: [{speech_16k.min():.4f}, {speech_16k.max():.4f}]")
        
        # 使用whisper的log mel spectrogram
        feat_whisper = whisper.log_mel_spectrogram(speech_16k, n_mels=128)
        print(f"Whisper mel shape: {feat_whisper.shape}")
        print(f"Whisper mel range: [{feat_whisper.min():.4f}, {feat_whisper.max():.4f}]")
        
        # 确保数据在正确的设备上并转换为numpy，添加batch维度
        feat_whisper_np = feat_whisper.detach().cpu().numpy()
        feat_length = np.array([feat_whisper.shape[2]], dtype=np.int32)
        
        print(f"Input to speech tokenizer:")
        print(f"  - feat shape: {feat_whisper_np.shape}")
        print(f"  - feat range: [{feat_whisper_np.min():.4f}, {feat_whisper_np.max():.4f}]")
        print(f"  - length: {feat_length}")
        
        # 运行speech tokenizer
        speech_token = speech_tokenizer_session.run(None, {
            speech_tokenizer_session.get_inputs()[0].name: feat_whisper_np,
            speech_tokenizer_session.get_inputs()[1].name: feat_length
        })[0]
        speech_token = speech_token.squeeze(0)
        mel = mel.transpose(0, 1)
        token_mel_ratio = 2
        # trim to align speech_token and speech_feat
        token_len = int(min(mel.shape[0] / token_mel_ratio, speech_token.shape[0]))
        speech_token = speech_token[:token_len]
        mel = mel[:token_len*token_mel_ratio]
        print(f"Speech tokenizer output shape: {speech_token.shape}")
        print(f'mel shape: {mel.shape}')
        mel_list.append(mel)
        speech_token_list.append(speech_token)
        texts_list.append(data['json']['text'])
    return mel_list,embedding_list,speech_token_list,texts_list

def create_batch_data(mel_list, embedding_list, speech_token_list, texts_list):
    """
    将处理后的数据整理成batch格式
    """
    from torch.nn.utils.rnn import pad_sequence
    
    batch_size = len(mel_list)
    
    # 1. 处理speech_feat (mel_list) - 需要padding
    feat_lengths = [mel.shape[1] for mel in mel_list]  # 获取每个样本的时间维度长度
    
    # 使用pad_sequence进行padding - 保持[batch, time, mel_dim]格式
    mel_sequences = [mel for mel in mel_list]  # [time, mel_dim]
    speech_feat = pad_sequence(mel_sequences, batch_first=True, padding_value=0).to(device)  # [batch, time, mel_dim]
    
    # 创建speech_feat_len tensor
    speech_feat_len = torch.tensor(feat_lengths, dtype=torch.long, device=device)
    
    # 2. 处理speech_token - 需要padding
    token_lengths = [len(tokens) for tokens in speech_token_list]
    
    # 使用pad_sequence进行padding
    token_sequences = [torch.tensor(tokens, dtype=torch.long) for tokens in speech_token_list]
    speech_token = pad_sequence(token_sequences, batch_first=True, padding_value=0).to(device)
    
    # 创建speech_token_len tensor
    speech_token_len = torch.tensor(token_lengths, dtype=torch.long, device=device)
    
    # 3. 处理embedding - 直接stack (所有embedding长度相同)
    embedding = torch.stack([torch.tensor(emb, dtype=torch.bfloat16) for emb in embedding_list], dim=0).to(device)
    
    # 验证 embedding 维度
    expected_embedding_dim = 192  # campplus embedding 维度
    if embedding.shape[1] != expected_embedding_dim:
        raise ValueError(f"Embedding dimension mismatch! Expected {expected_embedding_dim}, got {embedding.shape[1]}")
    
    # 4. 处理texts
    texts = texts_list
    
    return {
        'speech_feat': speech_feat.to(torch.bfloat16),           # [B, time, mel_dim]
        'speech_feat_len': speech_feat_len,   # [B]
        'speech_token': speech_token,         # [B, max_token_len]
        'speech_token_len': speech_token_len, # [B]
        'embedding': embedding,               # [B, embedding_dim] - 已经是 bfloat16
        'texts': texts                       # [B]
    }

# 使用自定义collate函数创建DataLoader
flow_model = configs['flow']
flow_model = flow_model.to(torch.bfloat16).to(device)
flow_model.train()
from torch.optim.adamw import AdamW
optimizer = AdamW(flow_model.parameters(), lr=1e-4)


data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=simple_collate_fn)
torch_device = torch.device(device)
for data in data_loader:
    print(f"Batch size: {len(data)}")
    mel_list,embedding_list,speech_token_list,texts_list = process_one_batch(data,feat_extractor,campplus_session,speech_tokenizer_session,sample_rate)
    
    # 创建格式化的batch数据
    batch_data = create_batch_data(mel_list, embedding_list, speech_token_list, texts_list)
    
    print("\n=== Batch Data Summary ===")
    print(f"speech_feat shape: {batch_data['speech_feat'].shape}")
    print(f"speech_feat_len: {batch_data['speech_feat_len']}")
    print(f"speech_token shape: {batch_data['speech_token'].shape}")
    print(f"speech_token_len: {batch_data['speech_token_len']}")
    print(f"embedding shape: {batch_data['embedding'].shape}")
    print(f"texts: {batch_data['texts']}")
    
    # 验证数据正确性
    print("\n=== Data Validation ===")
    for i in range(len(batch_data['texts'])):
        print(f"Sample {i}:")
        print(f"  - Original feat length: {mel_list[i].shape[1]}")
        print(f"  - Batch feat length: {batch_data['speech_feat_len'][i]}")
        print(f"  - Original token length: {len(speech_token_list[i])}")
        print(f"  - Batch token length: {batch_data['speech_token_len'][i]}")
        print(f"  - Text: {batch_data['texts'][i][:50]}...")
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):  
        loss = flow_model(batch_data,torch_device)
    print(loss)
    loss = loss['loss']
    loss.backward()
    optimizer.step()
    print(f'loss: {loss.item()}')
    # 现在你可以使用这些数据来训练模型
    # token = batch_data['speech_token']  # [B, max_token_len]
    # token_len = batch_data['speech_token_len']  # [B]
    # x1 = batch_data['speech_feat']  # [B, mel_dim, max_time] - This is X_1, the ground truth
    # feat_len = batch_data['speech_feat_len']  # [B]
    # embedding = batch_data['embedding']  # [B, embedding_dim]
    
    break




