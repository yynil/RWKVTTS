import click
import os
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
from torch.nn import Embedding
import shutil
@click.command()
@click.option("--input-path", type=click.Path(exists=True), required=True)
@click.option("--output-path", type=click.Path(exists=False), required=True)
@click.option("--enlarge_by", type=int, required=True)
def main(input_path, output_path, enlarge_by):
    """
    Enlarge the text vocabulary of a SparkLLM model.
    """
    print(f"Enlarging the text vocabulary of the model at {input_path}...")
    model = AutoModelForCausalLM.from_pretrained(input_path,trust_remote_code=True).cpu()
    original_tokenizer = AutoTokenizer.from_pretrained(input_path,trust_remote_code=True)
    original_vocab_size = original_tokenizer.vocab_size
    print(f"Original vocabulary size: {original_vocab_size}")
    new_vocab_size = original_vocab_size + enlarge_by
    print(f"New vocabulary size: {new_vocab_size}")
    new_model = AutoModelForCausalLM.from_pretrained(input_path,trust_remote_code=True).cpu()
    text_embedder = model.text_embedder
    actual_vocab_size = text_embedder.weight.shape[0]
    print(f"实际的text_embedder词汇表大小: {actual_vocab_size}")
    
    # 使用实际的词汇表大小而不是tokenizer的词汇表大小
    new_model.config.text_vocab_size = new_vocab_size
    new_text_embedder = Embedding(new_vocab_size, text_embedder.embedding_dim)
    new_text_embedder.weight.data[:actual_vocab_size] = text_embedder.weight.data
    new_model.text_embedder = new_text_embedder
    new_model.save_pretrained(output_path)
    print(f"New model saved to {output_path}")
    print(new_model)
    new_tokenizer = AutoTokenizer.from_pretrained(input_path,trust_remote_code=True)
    
    # 为 new_tokenizer 增加 enlarge_by 个新 token
    new_tokens = [f"SPCT_{i}" for i in range(enlarge_by)]
    new_tokenizer.add_tokens(new_tokens)
    new_tokenizer.save_pretrained(output_path)
    print(f"Added {enlarge_by} new tokens to tokenizer: {new_tokens}")
    
    #copy *py from input_path to output_path
    for file in os.listdir(input_path):
        if file.endswith(".py") and file != "enlarge_sparkllm_textvocab.py":
            shutil.copy(os.path.join(input_path, file), os.path.join(output_path, file))
    print(f"Copied {len(os.listdir(input_path))} files from {input_path} to {output_path}")

    #copy *txt from input_path to output_path
    for file in os.listdir(input_path):
        if file.endswith(".txt"):
            shutil.copy(os.path.join(input_path, file), os.path.join(output_path, file))
    print(f"Copied {len(os.listdir(input_path))} files from {input_path} to {output_path}")

    print(f"New model saved to {output_path}")

    #verify the new model weights and tokenizer
    print("\n=== 验证新模型和tokenizer ===")
    
    # 重新加载新模型和tokenizer进行验证
    loaded_model = AutoModelForCausalLM.from_pretrained(output_path, trust_remote_code=True).cpu()
    loaded_tokenizer = AutoTokenizer.from_pretrained(output_path, trust_remote_code=True)
    
    # 验证模型配置
    print(f"加载模型的词汇表大小: {loaded_model.config.text_vocab_size}")
    print(f"期望的词汇表大小: {new_vocab_size}")
    assert loaded_model.config.text_vocab_size == new_vocab_size, f"模型词汇表大小不匹配: {loaded_model.config.text_vocab_size} != {new_vocab_size}"
    
    # 验证tokenizer词汇表大小
    print(f"加载tokenizer的词汇表大小: {loaded_tokenizer.vocab_size}")
    assert loaded_tokenizer.vocab_size == new_vocab_size, f"Tokenizer词汇表大小不匹配: {loaded_tokenizer.vocab_size} != {new_vocab_size}"
    
    # 验证新添加的token
    for i in range(enlarge_by):
        token_name = f"SPCT_{i}"
        if token_name in loaded_tokenizer.get_vocab():
            token_id = loaded_tokenizer.get_vocab()[token_name]
            print(f"Token '{token_name}' 的ID: {token_id}")
            assert token_id >= original_vocab_size, f"新token的ID应该大于等于实际词汇表大小: {token_id} < {original_vocab_size}"
        else:
            raise AssertionError(f"新token '{token_name}' 未在tokenizer中找到")
    
    # 验证text_embedder权重
    loaded_text_embedder = loaded_model.text_embedder
    print(f"加载模型的text_embedder形状: {loaded_text_embedder.weight.shape}")
    print(f"期望的text_embedder形状: ({new_vocab_size}, {text_embedder.embedding_dim})")
    assert loaded_text_embedder.weight.shape == (new_vocab_size, text_embedder.embedding_dim), f"text_embedder权重形状不匹配"
    
    # 验证原始token的权重是否保持不变
    original_weights = text_embedder.weight.data
    loaded_original_weights = loaded_text_embedder.weight.data[:actual_vocab_size]
    weight_diff = torch.abs(original_weights - loaded_original_weights).max()
    print(f"原始token权重最大差异: {weight_diff}")
    assert weight_diff < 1e-6, f"原始token权重发生变化，最大差异: {weight_diff}"
    
    print("✅ 验证通过！新模型和tokenizer与期望一致。")
if __name__ == "__main__":
    main()