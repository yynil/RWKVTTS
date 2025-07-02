import torch
import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM
from model.llm.xy_llm import RWKV7XYLM

def verify_models(original_path, converted_path):
    """
    Verifies that the converted model correctly incorporates the weights of the original model.
    """
    print("--- Loading models for verification ---")
    try:
        original_model = AutoModelForCausalLM.from_pretrained(original_path, trust_remote_code=True)
        converted_model = AutoModelForCausalLM.from_pretrained(converted_path, trust_remote_code=True)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please ensure that 'xy_llm.py' is in the converted model directory and your PYTHONPATH is set correctly.")
        return

    all_ok = True
    original_vocab_size = original_model.config.vocab_size

    # 1. Verify the RWKV7Model backbone (excluding embeddings)
    print("\n--- 1. Verifying RWKV7Model Backbone ---")
    original_backbone_sd = original_model.model.state_dict()
    converted_backbone_sd = converted_model.model.state_dict()
    
    for key in original_backbone_sd:
        if key == 'embeddings.weight':
            print(f"Skipping '{key}' (handled separately)...")
            continue
        
        if key not in converted_backbone_sd:
            print(f"[MISMATCH] Key '{key}' not found in converted model backbone.")
            all_ok = False
            continue

        if not torch.allclose(original_backbone_sd[key], converted_backbone_sd[key]):
            print(f"[MISMATCH] Backbone weights for '{key}' do not match.")
            all_ok = False
        else:
            print(f"[OK] Backbone weights for '{key}' match.")

    # 2. Verify Channel 0 (Text) Embeddings and Head
    print("\n--- 2. Verifying Channel 0 (Text) Weights ---")
    # Embedding comparison
    original_emb = original_model.model.embeddings.weight
    converted_emb_ch0 = converted_model.embs[0].weight
    if not torch.allclose(original_emb, converted_emb_ch0[:original_vocab_size, :]):
        print("[MISMATCH] Channel 0 embedding weights do not match original.")
        all_ok = False
    else:
        print("[OK] Channel 0 embedding weights match original.")

    # Head comparison
    original_head = original_model.lm_head.weight
    converted_head_ch0 = converted_model.heads[0].weight
    if not torch.allclose(original_head, converted_head_ch0[:original_vocab_size, :]):
        print("[MISMATCH] Channel 0 head weights do not match original.")
        all_ok = False
    else:
        print("[OK] Channel 0 head weights match original.")

    # Final Summary
    print("\n--- Verification Summary ---")
    if all_ok:
        print("\033[92mAll checks passed! The converted model seems to be correct.\033[0m")
    else:
        print("\033[91mVerification failed. Some weights were not copied correctly.\033[0m")

def main():
    parser = argparse.ArgumentParser(description="Verify the conversion of an RWKV7 model to RWKV7XYLM.")
    parser.add_argument("--original_path", type=str, default="/home/yueyulin/models/rwkv7-0.4B-g1", help="Path to the original RWKV7 model directory.")
    parser.add_argument("--converted_path", type=str, default="/home/yueyulin/models/rwkv7-xy-0.4B-g1", help="Path to the converted RWKV7XYLM model directory.")
    args = parser.parse_args()

    verify_models(args.original_path, args.converted_path)

if __name__ == '__main__':
    main()
