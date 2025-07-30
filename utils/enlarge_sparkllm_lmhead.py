import click
import os
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
from torch.nn import Embedding,Linear
import shutil
@click.command()
@click.option("--input-path", type=click.Path(exists=True), required=True)
@click.option("--output-path", type=click.Path(exists=False), required=True)
def main(input_path, output_path):
    """
    Enlarge the text vocabulary of a SparkLLM model.
    """
    print(f"Enlarging the text vocabulary of the model at {input_path}...")
    model = AutoModelForCausalLM.from_pretrained(input_path,trust_remote_code=True).cpu()
    print(f"Original model: {model}")
    tokenizer = AutoTokenizer.from_pretrained(input_path,trust_remote_code=True)
    original_vocab_size = model.config.vocab_size
    print(f"Original vocabulary size: {original_vocab_size}")
    audio_global_vocab_size = model.config.audio_global_vocab_size
    new_vocab_size = original_vocab_size + audio_global_vocab_size
    print(f"New vocabulary size: {new_vocab_size}")
    new_model = AutoModelForCausalLM.from_pretrained(input_path,trust_remote_code=True).cpu()
    print(f"New model: {new_model}")
    lm_head = model.lm_head
    
    #enlarge the lm_head to the new_vocab_size
    new_lm_head = Linear(model.config.hidden_size, new_vocab_size, bias=False)
    new_lm_head.weight.data[:original_vocab_size] = lm_head.weight.data
    new_model.lm_head = new_lm_head
    new_model.config.vocab_size = new_vocab_size
    print(f"New model config: {new_model.config}")
    print(f"New model lm_head: {new_model.lm_head.weight.shape}")

    new_embeddings = Embedding(new_vocab_size, model.config.hidden_size)
    new_embeddings.weight.data[:original_vocab_size] = model.model.embeddings.weight.data
    new_model.model.embeddings = new_embeddings
    new_model.save_pretrained(output_path)
    #save tokenizer
    tokenizer.save_pretrained(output_path)

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

    #verify the new model
    new_model = AutoModelForCausalLM.from_pretrained(output_path,trust_remote_code=True).cpu()
    assert new_model.config.vocab_size == new_vocab_size
    assert new_model.lm_head.weight.shape[0] == new_vocab_size
    #compare the new model lm_head with the original model lm_head
    assert torch.allclose(new_model.lm_head.weight[:original_vocab_size], model.lm_head.weight)
    assert torch.allclose(new_model.model.embeddings.weight[:original_vocab_size], model.model.embeddings.weight)
    print(f"New model verified")

if __name__ == "__main__":
    main()