import torch
import torch.nn as nn

def train_step(model,batch):
    batch = {k: v.to(model.device) for k, v in batch.items()}
    output = model(batch)
    return output

def alter_emb_and_head(model, vocab_size, audio_token_size):
    old_embeddings = model.model.embeddings
    if vocab_size < model.config.vocab_size:
        print(f'No need to enlarge the vocabulary size: {model.config.vocab_size}')
    
    # 创建并初始化新的 embedding 层
    print(f'Enlarging vocabulary size from {model.config.vocab_size} to {vocab_size}')
    embedding_dim = old_embeddings.weight.size(1)
    current_vocab_size = old_embeddings.weight.size(0)
    new_embeddings = nn.Embedding(vocab_size, embedding_dim)
    with torch.no_grad():
        new_embeddings.weight[:current_vocab_size, :] = old_embeddings.weight.data
        std = old_embeddings.weight.std().item()
        new_embeddings.weight[current_vocab_size:, :].normal_(mean=0.0, std=std)
    model.model.embeddings = new_embeddings
    model.config.vocab_size = vocab_size
    # old_head = model.lm_head
    head_dim = model.config.hidden_size
    new_head = nn.Linear(head_dim, audio_token_size+1)
    with torch.no_grad():
        #init the new head with random values
        new_head.weight.normal_(mean=0.0, std=0.02)
    model.lm_head = new_head  
    print(f'Enlarging head size from {head_dim} to {audio_token_size}')
    return model