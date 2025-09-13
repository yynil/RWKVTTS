import torch
from argparse import Namespace
def parser_config_from_checkpoint(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    args = Namespace()
    args.n_head, args.head_size = state_dict['blocks.0.att.r_k'].shape
    args.head_size_a = args.head_size
    args.head_size_divisor = 1
    args.vocab_size , args.n_embd = state_dict['emb.weight'].shape
    args.n_layer = 0
    keys = list(state_dict.keys())
    for key in keys:
        layer_id = int(key.split('.')[1]) if ('blocks.' in key) else 0
        args.n_layer = max(args.n_layer, layer_id+1)
    args.need_init_tmix = False
    args.need_init_cmix = False
    args.dropout = 0
    args.grad_cp = 0
    return args


if __name__ == "__main__":
    args = parser_config_from_checkpoint("/home/yueyulin/models/rwkv7-g1-0.4b-20250324-ctx4096.pth")
    print(args)