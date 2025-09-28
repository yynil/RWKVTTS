from data.spark.multiple_webdataset import MultipleWebDataset
from data.utils.collator import xy_data_collator
import os
import glob
from transformers import AutoTokenizer
from XY_Tokenizer.xy_tokenizer.model import XY_Tokenizer
import torch
from functools import partial
if __name__ == '__main__':
    data_dir = '/external_data/voxbox_wids/aishell-3/'
    data_files = glob.glob(os.path.join(data_dir, '*.tar'))
    dataset = MultipleWebDataset(
        data_files=data_files,
        target_sr=16000,
        target_channels=1,
        shuffle=False,
        verify_tar=False
    )
    print(dataset[0])
    print(dataset[1])
    print(dataset[2])
    print(dataset[3])

    model_dir = '/home/yueyulin/models/rwkv7-xy-0.4B-g1/'
    tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
    config_file = 'third_party/XY_Tokenizer/config/xy_tokenizer_config.yaml'
    ckpt_file = '/home/yueyulin/models/XY_Tokenizer_TTSD_V0/xy_tokenizer.ckpt'
    xy_tokenizer = XY_Tokenizer.load_from_checkpoint(config_path=config_file,ckpt_path=ckpt_file)
    print(xy_tokenizer)
    xy_tokenizer.eval()
    xy_tokenizer.to('cuda')
    wav_list = [torch.from_numpy(dataset[0]['audio']['array']),torch.from_numpy(dataset[1]['audio']['array']),torch.from_numpy(dataset[2]['audio']['array']),torch.from_numpy(dataset[3]['audio']['array'])]
    output = xy_tokenizer.encode(wav_list)
    output = output['codes_list']
    print(output)
    for i in range(len(output)):
        print(output[i].shape)
        print(output[i])
    text_shift_size = 65536
    speech_vocab_size = 1025
    collate_fn = partial(xy_data_collator,text_tokenizer=tokenizer,xy_tokenizer=xy_tokenizer,num_channels=8,text_shift_size=text_shift_size,speech_vocab_size=speech_vocab_size,device='cuda')
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset,batch_size=2,collate_fn=collate_fn)
    for batch in dataloader:
        print('---Input IDs---')
        print(batch['input_ids'].tolist())
        print('---Labels---')
        print(batch['labels'].tolist())
        print('---Attention Mask---')
        print(batch['attention_mask'].tolist())
        break