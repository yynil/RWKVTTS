from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Model
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM,RWKV7Config
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    audio_lm_config = {
        "a_low_rank_dim": 64,
        "attn": None,
        "attn_mode": "chunk",
        "bos_token_id": 0,
        "decay_low_rank_dim": 64,
        "eos_token_id": 0,
        "fuse_cross_entropy": True,
        "fuse_norm": False,
        "gate_low_rank_dim": 128,
        "head_dim": 64,
        "hidden_act": "sqrelu",
        "hidden_ratio": 4.0,
        "hidden_size": 768,
        "initializer_range": 0.006,
        "intermediate_size": 3072,
        "max_position_embeddings": 2048,
        "model_type": "rwkv7",
        "norm_bias": True,
        "norm_eps": 1e-05,
        "norm_first": True,
        "num_heads": 32,
        "num_hidden_layers": 12,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "transformers_version": "4.48.0",
        "use_cache": True,
        "v_low_rank_dim": 32,
        "vocab_size": 4096+8193
        }
    audio_lm_config = RWKV7Config.from_dict(audio_lm_config)
    print(f'init audio_lm')
    audio_lm = RWKV7ForCausalLM(audio_lm_config)
    print(f'save audio_lm to {args.output_path}')
    audio_lm.save_pretrained(args.output_path)
    print(f'done')

    print(f'Try to reload audio_lm')
    audio_lm = RWKV7ForCausalLM.from_pretrained(args.output_path)
    print(f'audio_lm: {audio_lm}')
    print(f'done')