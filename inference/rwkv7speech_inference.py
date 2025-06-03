from typing import List
import torch
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from pathlib import Path
import re
import soundfile as sf
def load_global_tokens(audio_tokenizer, directory,device:str):
    """
    Args:
        audio_tokenizer: 
        directory: 
    """
    ref_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                ref_files.append(Path(os.path.join(root, file)))
   
    print(f'load global tokens from {ref_files}')
    all_global_tokens = []
    all_global_tokens_ids = []
    characters = []
    for ref_file in ref_files:
        parent_dir = os.path.dirname(ref_file)
        global_tokens_ids, semantic_tokens = audio_tokenizer.tokenize(ref_file)
        global_tokens_ids = global_tokens_ids.squeeze(0).squeeze(0).detach().cpu().tolist()
        global_tokens ="".join([f"<|bicodec_global_{token}|>" for token in global_tokens_ids])
        all_global_tokens.append(global_tokens)
        all_global_tokens_ids.append(global_tokens_ids)
        characters.append(parent_dir.split('/')[-1])
    return all_global_tokens, all_global_tokens_ids,characters

def create_inputs(texts: List[str],global_tokens_ids: List[List[int]],semantic_tokens_ids: List[List[int]],tokenizer,llm,pad_token_id=0):
    """
    Args:
        texts: List[str]
        global_tokens_ids: List[List[int]]
        semantic_tokens_ids: List[List[int]]
    """
    input_ids_embs_list = []
    assert len(texts) == len(global_tokens_ids) == len(semantic_tokens_ids), \
        f"输入列表长度不一致: texts({len(texts)}), global_tokens_ids({len(global_tokens_ids)}), semantic_tokens_ids({len(semantic_tokens_ids)})"
    max_length = 0
    B = len(texts)
    for text,global_tokens_id,semantic_tokens_id in zip(texts,global_tokens_ids,semantic_tokens_ids):
        input_ids = tokenizer.encode(text)
        input_ids_embs = llm.text_embedder(torch.tensor([input_ids],dtype=torch.long,device=llm.device))#1,len_of_input_ids,C
        global_tokens_embs = llm.global_embedder(torch.tensor([global_tokens_id],dtype=torch.long,device=llm.device))#1,32,C
        semantic_tokens_embs = llm.model.embeddings(torch.tensor([semantic_tokens_id],dtype=torch.long,device=llm.device))#1,len_of_semantic_tokens_ids,C
        tts_tag_embedder_0 = llm.tts_tag_embedder(torch.tensor([[0]],dtype=torch.long,device=llm.device))#1,C
        tts_tag_embedder_1 = llm.tts_tag_embedder(torch.tensor([[1]],dtype=torch.long,device=llm.device))#1,C
        input_ids_embs = torch.cat([input_ids_embs,tts_tag_embedder_0,global_tokens_embs,tts_tag_embedder_1,semantic_tokens_embs],dim=1)
        input_ids_embs_list.append(input_ids_embs)
        max_length = max(max_length,input_ids_embs.shape[1])
    #left padding and create attention mask
    attention_mask = torch.zeros(B,max_length,dtype=torch.long,device=llm.device)
    for i in range(B):
        attention_mask[i,-input_ids_embs_list[i].shape[1]:] = 1
        embs_pad_t = max_length - input_ids_embs_list[i].shape[1]
        #left padding input_ids_embs_list[i]
        input_ids_embs_list[i] = torch.cat([torch.zeros(1,embs_pad_t,input_ids_embs_list[i].shape[2],dtype=torch.long,device=llm.device),input_ids_embs_list[i]],dim=1)

    input_ids_embs = torch.cat(input_ids_embs_list,dim=0)
    return input_ids_embs,attention_mask

def compose_input(input_ids,attention_mask,global_tokens_ids,semantic_tokens_ids,llm):
    """
    The tts input is composed as the following format:
    0. input_ids_embs: converted by llm.text_embedder(input_ids)
    1. llm.tts_tag_embedder(0) : start of global tokens
    2. llm.global_embedder(global_tokens_ids)
    3. llm.tts_tag_embedder(1) : end of global tokens
    4. llm.model.embeddings(semantic_tokens_ids): prompted audio tokens
    """
    B,T = input_ids.shape
    input_ids_embs = llm.text_embedder(input_ids)#B,T,C
    global_tokens_embs = llm.global_embedder(global_tokens_ids)#B,32,C
    semantic_tokens_embs = llm.model.embeddings(semantic_tokens_ids)#B,T,C
    tts_tag_embedder_0 = llm.tts_tag_embedder(torch.tensor(0).to(device))#1,C
    #repeat tts_tag_embedder_0 tts_tag_embedder_1 to B,1,C
    tts_tag_embedder_0 = tts_tag_embedder_0.repeat(B,1,1)
    tts_tag_embedder_1 = llm.tts_tag_embedder(torch.tensor(1).to(device))#1,C
    tts_tag_embedder_1 = tts_tag_embedder_1.repeat(B,1,1)
    _,semantic_T = semantic_tokens_ids.shape
    _,global_T = global_tokens_ids.shape
    input_ids_embs = torch.cat([input_ids_embs,tts_tag_embedder_0,global_tokens_embs,tts_tag_embedder_1,semantic_tokens_embs],dim=1)
    #left padding's attention mask 
    attention_mask = torch.cat([attention_mask,torch.ones(B,1,dtype=torch.long).to(device),torch.ones(B,global_T,dtype=torch.long).to(device),torch.ones(B,1,dtype=torch.long).to(device),torch.ones(B,semantic_T,dtype=torch.long).to(device)],dim=1)
    return input_ids_embs,attention_mask

if __name__ == "__main__":
    model_dir = '/home/yueyulin/models/Spark-TTS-0.5B/'
    demo_dir = 'demos'
    llm_dir = '/home/yueyulin/models/rwkv7-0.1B-g1-respark-speech'
    device = 'cuda:3'
    llm = AutoModelForCausalLM.from_pretrained(llm_dir,trust_remote_code=True).bfloat16().to(device)
    print(llm)
    llm.eval()
    tokenizer = AutoTokenizer.from_pretrained(llm_dir,trust_remote_code=True)
    print(tokenizer)
    text = ['Hello, how are you?','I am fine, thank you. I am your daddy.']
    model_inputs = tokenizer(text,return_tensors='pt',padding=True,truncation=True,max_length=1024,padding_side='left')
    print(model_inputs)
    audio_tokenizer = BiCodecTokenizer(model_dir,device='cpu')
    all_global_tokens,all_global_tokens_ids,characters = load_global_tokens(audio_tokenizer,demo_dir,device)
    print(all_global_tokens)
    print(all_global_tokens_ids)
    print(characters)

    input_ids = model_inputs['input_ids'].to(device)
    attention_mask = model_inputs['attention_mask'].to(device)
    semantic_tokens_ids = torch.randint(0,8192,size=(input_ids.shape[0],10))
    global_tokens_ids = torch.tensor([all_global_tokens_ids[0],all_global_tokens_ids[1]]).to(device)
    semantic_tokens_ids = semantic_tokens_ids.to(device)
    input_ids_embs,attention_mask = compose_input(input_ids,attention_mask,global_tokens_ids,semantic_tokens_ids,llm)
    
    print(input_ids_embs.shape)
    print(attention_mask.shape)
    print(input_ids_embs)
    print(attention_mask)
    _,input_len,_ = input_ids_embs.shape
    print(input_len)
    print(llm.config.vocab_size)
    with torch.no_grad():
        output = llm.generate(inputs_embeds=input_ids_embs,
                              attention_mask=attention_mask,
                              max_new_tokens=1024,
                              do_sample=True,
                              top_k=50,
                              top_p=0.95,
                              pad_token_id=llm.config.vocab_size-1,
                              eos_token_id=llm.config.vocab_size-1,
                              )
        print(output.shape)
        print(output)
    print("========Test various inputs=========")
    global_tokens_ids = [[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]]
    semantic_tokens_ids = [[6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]]
    input_ids_embs,attention_mask = create_inputs(text,global_tokens_ids,semantic_tokens_ids,tokenizer,llm)
    print(input_ids_embs.shape)
    print(attention_mask.shape)
    print(input_ids_embs)
    print(attention_mask)

    with torch.no_grad():
        output = llm.generate(inputs_embeds=input_ids_embs,
                              attention_mask=attention_mask,
                              max_new_tokens=1024,
                              do_sample=True,
                              top_k=50,
                              top_p=0.95,
                              pad_token_id=llm.config.vocab_size-1,
                              eos_token_id=llm.config.vocab_size-1,
                              )
        print(output.shape)
        print(output)