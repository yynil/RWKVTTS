import torch
import sys
import safetensors

input_path = sys.argv[1]

# w = torch.load(input_path)

w = {}
with safetensors.safe_open(input_path, framework="pt") as f:
    for k in f.keys():
        w[k] = f.get_tensor(k)

w_new = {}
for k, v in w.items():
    k_orig = k
    k = k.replace('model.', '').replace('layers.', 'blocks.').replace('lm_head', 'head')
    k = k.replace('ffn_norm', 'ln2').replace('attn_norm', 'ln1').replace('pre_norm', 'ln0')
    k = k.replace('g_norm', 'ln_x')
    k = k.replace('norm', 'ln_out')
    k = k.replace('attn', 'att')
    k = k.replace('r_proj', 'receptance')
    k = k.replace('k_proj', 'key')
    k = k.replace('v_proj', 'value')
    k = k.replace('o_proj', 'output')
    if '_lora.lora.' in k and 'weight' in k:
        v = v.transpose(0, 1)
    k = k.replace('_lora.lora.2.bias', '0')
    k = k.replace('_lora.lora.2.weight', '2')
    k = k.replace('_lora.lora.0.weight', '1')

    if k == k_orig:
        print("untouched key: ", k)

    if 'att.x_x' in k:
        tensors = torch.split(v, 1, dim=0)
        names = ['r', 'w', 'k', 'v', 'a', 'g']
        for i in range(len(names)):
            w_new[k.replace('x_x', f'x_{names[i]}')] = tensors[i]
    else:
        w_new[k] = v

# print(w_new.keys())
# quit()

global_vocab_size = w_new['global_embedder.weight'].shape[0]
text_vocab_size = w_new['text_embedder.weight'].shape[0]
tts_tag_vocab_size = w_new['tts_tag_embedder.weight'].shape[0]
semantic_vocab_size = w_new['embeddings.weight'].shape[0]

# print(global_vocab_size, text_vocab_size, tts_tag_vocab_size, semantic_vocab_size)
# new embedding: | semantic 8193 | tts_tag 3 | global 4096 | text 65536 |

# del w_new['embeddings.weight']
w_new['emb.weight'] = torch.cat([w_new['embeddings.weight'], 
                                w_new['tts_tag_embedder.weight'], 
                                w_new['global_embedder.weight'], 
                                w_new['text_embedder.weight']], dim=0)

del w_new['text_embedder.weight']
del w_new['tts_tag_embedder.weight']
del w_new['global_embedder.weight']
del w_new['embeddings.weight']

print(w_new.keys())
# quit()

# print(w_new['emb.weight'].shape)
print(w_new['head.weight'].shape)
torch.save(w_new, sys.argv[1].replace('.safetensors', '_converted.pth'))

w_new['head.weight'] = torch.cat([w_new['head.weight'], torch.zeros((w_new['emb.weight'].shape[0]-w_new['head.weight'].shape[0], w_new['head.weight'].shape[1]))], dim=0)
print(w_new['head.weight'].shape)
torch.save(w_new, sys.argv[1].replace('.safetensors', '_padded.pth'))