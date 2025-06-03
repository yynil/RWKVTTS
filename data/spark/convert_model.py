from utils.utilities import get_respark_tts_tokenizer
from transformers import AutoModelForCausalLM
import torch
import os
import shutil
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/home/yueyulin/models/rwkv7-0.1B-g1")
    parser.add_argument("--output_dir", type=str, default="/home/yueyulin/models/rwkv7-0.1B-g1-respark-audio")
    args = parser.parse_args()
    tokenizer,original_vocab_size = get_respark_tts_tokenizer(args.model_dir)
    new_vocab_size = tokenizer.vocab_size
    print(f"Original vocab size: {original_vocab_size}")
    print(f"New vocab size: {new_vocab_size}")
    audio_vocab_size = new_vocab_size = original_vocab_size

    model = AutoModelForCausalLM.from_pretrained(args.model_dir,trust_remote_code=True)
    
    # 检查模型的实际词汇表大小
    model_vocab_size = model.model.embeddings.weight.shape[0]
    print(f"Model vocab size: {model_vocab_size}")
    
    if model_vocab_size != original_vocab_size:
        print(f"Warning: Model vocab size ({model_vocab_size}) doesn't match original vocab size ({original_vocab_size})")
        original_vocab_size = model_vocab_size
    
    # 扩展嵌入层
    old_embeddings = model.model.embeddings
    new_embeddings = torch.nn.Embedding(new_vocab_size, old_embeddings.embedding_dim)
    # 复制原始权重
    new_embeddings.weight.data[:original_vocab_size] = old_embeddings.weight.data
    # 初始化新增的嵌入向量
    if new_vocab_size > original_vocab_size:
        torch.nn.init.normal_(new_embeddings.weight.data[original_vocab_size:], mean=0.0, std=0.02)
    model.model.embeddings = new_embeddings
    
    # 扩展输出层
    old_lm_head = model.lm_head
    new_lm_head = torch.nn.Linear(old_lm_head.in_features, new_vocab_size, bias=False)
    # 复制原始权重
    new_lm_head.weight.data[:original_vocab_size] = old_lm_head.weight.data
    # 初始化新增的权重
    if new_vocab_size > original_vocab_size:
        torch.nn.init.normal_(new_lm_head.weight.data[original_vocab_size:], mean=0.0, std=0.02)
    model.lm_head = new_lm_head
    
    # 更新模型的配置
    model.config.vocab_size = new_vocab_size
    
    # 保存模型和分词器
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    # copy *py from model_dir to output_dir
    for file in os.listdir(args.model_dir):
        if file.endswith('.py'):
            shutil.copy(os.path.join(args.model_dir, file), os.path.join(args.output_dir, file))
            print(f"Copied {file} to {args.output_dir}")
    # copy *txt from model_dir to output_dir
    for file in os.listdir(args.model_dir):
        if file.endswith('.txt'):
            shutil.copy(os.path.join(args.model_dir, file), os.path.join(args.output_dir, file))
            print(f"Copied {file} to {args.output_dir}")
    # copy *json from model_dir to output_dir