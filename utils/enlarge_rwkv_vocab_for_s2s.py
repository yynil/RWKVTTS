import torch
import os
import click
from model.llm.rwkv_s2s import RWKV7S2S
from model.llm.rwkv_s2s_single_ffn import RWKV7S2S_SingleFFN
from argparse import Namespace

# args = {
#         "n_layer": 24,
#         "n_embd": 1024,
#         "vocab_size": 65536,
#         "head_size_a": 64,
#         "head_size_divisor": 1,
#         "n_layer": 24,
#         "dropout": 0.1,
#         "need_init_tmix": True,
#         "need_init_cmix": True,
#         "grad_cp": 1,
#         "audio_vocab_size": 8192,
#     }
def determin_args(state_dict) -> Namespace:
    args = Namespace()
    n_layer = 0
    for key, value in state_dict.items():
        if key.startswith("blocks."):
            index_of_dot = key.find(".", len("blocks."))
            current_layer = int(key[len("blocks."):index_of_dot])+1
            if current_layer > n_layer:
                n_layer = current_layer
    args.n_layer = n_layer
    args.n_embd = state_dict["emb.weight"].shape[1]
    args.vocab_size = state_dict["emb.weight"].shape[0]
    args.head_size_a = state_dict["blocks.0.att.r_k"].shape[1]
    args.head_size_divisor = 1
    args.text_vocab_size = args.vocab_size
    return args

@click.command()
@click.option("--input_path", type=str, required=False,default="/home/yueyulin/models/rwkv7-g1-0.4b-20250324-ctx4096.pth")
@click.option("--output_path", type=str, required=False,default="/home/yueyulin/models/rwkvs2s")
@click.option("--spct_size", type=int, default=100)
@click.option("--audio_vocab_size", type=int, default=8193)
@click.option("--audio_global_token_size", type=int, default=1024)
@click.option("--rwkv_vocab_file", type=str, default="tokenizer/rwkv_vocab_v20230424.txt")
@click.option("--single_ffn", is_flag=True, help="Export as single FFN version")
def enlarge_rwkv_vocab_for_s2s(input_path, output_path, spct_size, audio_vocab_size, audio_global_token_size, rwkv_vocab_file, single_ffn):
    os.makedirs(output_path, exist_ok=True)
    state_dict = torch.load(input_path)
    args = determin_args(state_dict)
    print(f"args determined by state_dict: {args}")
    args.audio_vocab_size = audio_vocab_size
    args.vocab_size = args.text_vocab_size + spct_size + audio_vocab_size + audio_global_token_size
    args.need_init_tmix = False
    args.need_init_cmix = False
    args.grad_cp = 0
    args.dropout = 0
    
    # 根据模式选择模型类型
    if single_ffn:
        print("Creating RWKV7S2S_SingleFFN model...")
        model = RWKV7S2S_SingleFFN(args)
    else:
        print("Creating RWKV7S2S model...")
        model = RWKV7S2S(args)
    
    print(model)
    
    # 1. 复制 block.* 权重到 audio_blocks.*，并转换为float32（仅对非single_ffn模式）
    audio_blocks_state_dict = {}
    if not single_ffn:
        for key, value in state_dict.items():
            if key.startswith("blocks."):
                audio_blocks_state_dict[key.replace("blocks.", "audio_blocks.")] = value.float()
    
    # 2. 扩展嵌入层权重
    new_emb = torch.nn.Embedding(args.vocab_size, args.n_embd)
    # 将原始权重转换为float32
    new_emb.weight.data[:args.text_vocab_size] = state_dict["emb.weight"].float()
    
    # 初始化新增词表的嵌入权重（使用较小的随机值）
    if args.text_vocab_size < args.vocab_size:
        # 使用原始嵌入权重的标准差来初始化新权重
        std = state_dict["emb.weight"].float().std()
        new_emb.weight.data[args.text_vocab_size:] = torch.randn(
            args.vocab_size - args.text_vocab_size, 
            args.n_embd, 
            dtype=torch.float32,
            device=state_dict["emb.weight"].device
        ) * std * 0.1  # 使用较小的初始化值
    
    state_dict["emb.weight"] = new_emb.weight
    
    # 3. 处理文本输出头权重
    if "head.weight" in state_dict:
        # 如果原始模型有head权重，需要扩展它
        original_head_weight = state_dict["head.weight"]
        new_head = torch.nn.Linear(args.n_embd, args.text_vocab_size, bias=False)
        # 将原始权重转换为float32
        new_head.weight.data[:original_head_weight.shape[0]] = original_head_weight.float()
        # 初始化新增部分
        if original_head_weight.shape[0] < args.text_vocab_size:
            std = original_head_weight.float().std()
            new_head.weight.data[original_head_weight.shape[0]:] = torch.randn(
                args.text_vocab_size - original_head_weight.shape[0],
                args.n_embd,
                dtype=torch.float32,
                device=original_head_weight.device
            ) * std * 0.1
        state_dict["head.weight"] = new_head.weight
    else:
        # 如果没有原始head权重，创建新的
        state_dict["head.weight"] = torch.nn.Linear(args.n_embd, args.text_vocab_size, bias=False).weight
    
    # 4. 初始化音频输出头权重
    audio_head = torch.nn.Linear(args.n_embd, audio_vocab_size, bias=False)
    state_dict["audio_head.weight"] = audio_head.weight
    
    # 对于single_ffn模式，不需要ln_out_audio，使用ln_out
    if single_ffn:
        if "ln_out.weight" not in state_dict:
            device = state_dict["emb.weight"].device
            state_dict["ln_out.weight"] = torch.ones(args.n_embd, dtype=torch.float32, device=device)
            state_dict["ln_out.bias"] = torch.zeros(args.n_embd, dtype=torch.float32, device=device)
    else:
        if "ln_out_audio.weight" not in state_dict:
            # 使用float32格式
            device = state_dict["emb.weight"].device
            state_dict["ln_out_audio.weight"] = torch.ones(args.n_embd, dtype=torch.float32, device=device)
            state_dict["ln_out_audio.bias"] = torch.zeros(args.n_embd, dtype=torch.float32, device=device)
    
    # 6. 将原始blocks权重也转换为float32
    for key, value in state_dict.items():
        if key.startswith("blocks."):
            state_dict[key] = value.float()
    
    # 7. 更新状态字典（仅对非single_ffn模式）
    if not single_ffn:
        state_dict.update(audio_blocks_state_dict)
    
    # 8. 加载权重到模型
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    
    # 9. 初始化audio_blocks的ffn参数（仅对非single_ffn模式）
    if not single_ffn:
        for i in range(args.n_layer):
            model.audio_blocks[i].ffn._init_params(args)
    
    # 10. 保存模型
    if single_ffn:
        model_filename = f"rwkv7_s2s_single_ffn_{args.n_layer}l_{args.n_embd}d_{args.text_vocab_size}t_{args.audio_vocab_size}a_{audio_global_token_size}g_{args.vocab_size}v_{args.head_size_a}h.pth"
    else:
        model_filename = f"rwkv7_s2s_{args.n_layer}l_{args.n_embd}d_{args.text_vocab_size}t_{args.audio_vocab_size}a_{audio_global_token_size}g_{args.vocab_size}v_{args.head_size_a}h.pth"
    
    torch.save(model.state_dict(), os.path.join(output_path, model_filename))
    
    # 11. 保存配置信息
    config = {
        "n_layer": args.n_layer,
        "n_embd": args.n_embd,
        "vocab_size": args.vocab_size,
        "text_vocab_size": args.text_vocab_size,
        "audio_vocab_size": args.audio_vocab_size,
        "spct_size": spct_size,
        "audio_global_token_size": audio_global_token_size,
        "head_size_a": args.head_size_a,
        "head_size_divisor": args.head_size_divisor,
        "dropout": args.dropout,
        "grad_cp": args.grad_cp,
        "model_type": "RWKV7S2S_SingleFFN" if single_ffn else "RWKV7S2S",
    }
    
    import json
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Model saved to {output_path}")
    print(f"Model type: {'RWKV7S2S_SingleFFN' if single_ffn else 'RWKV7S2S'}")
    print(f"Model filename: {model_filename}")
    print(f"Final vocab size: {args.vocab_size}")
    print(f"Text vocab size: {args.text_vocab_size}")
    print(f"Audio vocab size: {args.audio_vocab_size}")
    print(f"SPCT size: {spct_size}")
    print(f"Audio global token size: {audio_global_token_size}")

    #start to enlarge the vocab
    
    # Read the original vocab file
    original_vocab_file = rwkv_vocab_file
    output_vocab_file = os.path.join(output_path, "rwkv_vocab_enlarged.txt")
    
    # Read existing vocab to get the next available index
    with open(original_vocab_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Find the maximum index (assuming it's the last line)
    max_idx = 0
    for line in lines:
        if line.strip():
            idx = int(line[:line.index(' ')])
            max_idx = max(max_idx, idx)
    
    print(f"Original vocab size: {max_idx}")
    print(f"Adding new tokens starting from index: {max_idx + 1}")
    
    # Create new tokens
    new_tokens = []
    current_idx = max_idx + 1
    
    # Add SPCT tokens (SPCT_0 to SPCT_99)
    for i in range(spct_size):
        token_str = f"SPCT_{i}"
        token_bytes = token_str.encode("utf-8")
        new_tokens.append(f"{current_idx} '{token_str}' {len(token_bytes)}")
        current_idx += 1
    
    # Add SEMANTIC_TOKEN_ID tokens (SEMANTIC_TOKEN_ID_0 to SEMANTIC_TOKEN_ID_8192)
    for i in range(audio_vocab_size):
        token_str = f"SEMANTIC_TOKEN_ID_{i}"
        token_bytes = token_str.encode("utf-8")
        new_tokens.append(f"{current_idx} '{token_str}' {len(token_bytes)}")
        current_idx += 1
    
    # Add GLOBAL_TOKEN_ID tokens (GLOBAL_TOKEN_ID_0 to GLOBAL_TOKEN_ID_1023)
    for i in range(audio_global_token_size):
        token_str = f"GLOBAL_TOKEN_ID_{i}"
        token_bytes = token_str.encode("utf-8")
        new_tokens.append(f"{current_idx} '{token_str}' {len(token_bytes)}")
        current_idx += 1
    
    # Write the enlarged vocab file
    with open(output_vocab_file, "w", encoding="utf-8") as f:
        # Write original tokens
        f.writelines(lines)
        # Write new tokens
        for token_line in new_tokens:
            f.write(token_line + "\n")
    
    print(f"Enlarged vocab saved to: {output_vocab_file}")
    print(f"Added {len(new_tokens)} new tokens")
    print(f"Final vocab size: {current_idx - 1}")

if __name__ == "__main__":
    enlarge_rwkv_vocab_for_s2s()