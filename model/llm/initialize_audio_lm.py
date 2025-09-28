from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Config, RWKV7Model
from transformers.modeling_utils import _init_weights
audio_config = {
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
        "norm_bias": True,
        "norm_eps": 1e-05,
        "norm_first": True,
        "num_heads": 32,
        "num_hidden_layers": 12,
        "tie_word_embeddings": False,
        "use_cache": True,
        "v_low_rank_dim": 32,
        "vocab_size": 12288
    }
audio_config_obj = RWKV7Config(**audio_config)
print("Initializing audio_lm_model...")
audio_lm_model = RWKV7Model(audio_config_obj)
print(audio_lm_model)
audio_lm_model.save_pretrained("/home/yueyulin/models/rwkv7_0.1b_audio_lm_12k_vocab")
print("audio_lm_model saved")