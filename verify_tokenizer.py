import argparse
from transformers import AutoTokenizer

def verify_tokenizers(original_path, converted_path, speech_vocab_size):
    """
    Verifies that the new tokenizer is a valid superset of the original.
    """
    print("--- Loading tokenizers for verification ---")
    try:
        original_tokenizer = AutoTokenizer.from_pretrained(original_path, trust_remote_code=True)
        converted_tokenizer = AutoTokenizer.from_pretrained(converted_path, trust_remote_code=True)
        print("Tokenizers loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizers: {e}")
        return

    all_ok = True
    original_vocab_size = len(original_tokenizer)
    converted_vocab_size = len(converted_tokenizer)

    print(f"\n[INFO] Original vocab size: {original_vocab_size}")
    print(f"[INFO] Converted vocab size: {converted_vocab_size}")

    # 1. Verify the original vocabulary part
    print("\n--- 1. Verifying Consistency of Original Vocabulary ---")
    mismatched_tokens = 0
    for i in range(original_vocab_size):
        original_token = original_tokenizer.convert_ids_to_tokens(i)
        converted_token = converted_tokenizer.convert_ids_to_tokens(i)
        if original_token != converted_token:
            all_ok = False
            mismatched_tokens += 1
            if mismatched_tokens < 10: # Print first few mismatches
                 print(f"[MISMATCH] ID {i}: Original is '{original_token}', Converted is '{converted_token}'")
    
    if mismatched_tokens == 0:
        print("[OK] All tokens in the original vocabulary range are consistent.")
    else:
        print(f"[MISMATCH] Found {mismatched_tokens} inconsistent tokens in the original vocabulary range.")

    # 2. Verify the expanded vocabulary part
    print("\n--- 2. Verifying Correctness of Expanded Vocabulary ---")
    # Generate the list of expected added tokens in the correct order
    special_tokens = [f"[S{i}]" for i in range(10)] + [f"[CTL{i}]" for i in range(90)]
    speech_tokens = [f"[SP{i}]" for i in range(speech_vocab_size)]
    expected_added_tokens = special_tokens + speech_tokens

    expected_vocab_size = original_vocab_size + len(expected_added_tokens)
    if converted_vocab_size != expected_vocab_size:
        print(f"[MISMATCH] Converted vocab size is {converted_vocab_size}, but expected {expected_vocab_size}.")
        all_ok = False
    else:
        print("[OK] Converted vocab size matches expected size.")

        mismatched_added_tokens = 0
        for i, expected_token in enumerate(expected_added_tokens):
            actual_token_id = original_vocab_size + i
            actual_token = converted_tokenizer.convert_ids_to_tokens(actual_token_id)
            if actual_token != expected_token:
                all_ok = False
                mismatched_added_tokens += 1
                if mismatched_added_tokens < 10:
                    print(f"[MISMATCH] ID {actual_token_id}: Expected '{expected_token}', but found '{actual_token}'")
        
        if mismatched_added_tokens == 0:
            print("[OK] All new special and speech tokens are correct and in order.")
        else:
            print(f"[MISMATCH] Found {mismatched_added_tokens} incorrect new tokens.")

    # Final Summary
    print("\n--- Verification Summary ---")
    if all_ok:
        print("\033[92mAll checks passed! The new tokenizer is a valid superset of the original.\033[0m")
    else:
        print("\033[91mTokenizer verification failed. Please review the mismatches above.\033[0m")

def main():
    parser = argparse.ArgumentParser(description="Verify the conversion of a RWKV7 tokenizer.")
    parser.add_argument("--original_path", type=str, default="/home/yueyulin/models/rwkv7-0.4B-g1", help="Path to the original RWKV7 model directory.")
    parser.add_argument("--converted_path", type=str, default="/home/yueyulin/models/rwkv7-xy-0.4B-g1", help="Path to the converted RWKV7XYLM model directory.")
    parser.add_argument("--speech_vocab_size", type=int, default=1024, help="Vocabulary size for the speech channels used during conversion.")
    args = parser.parse_args()

    verify_tokenizers(args.original_path, args.converted_path, args.speech_vocab_size)

if __name__ == '__main__':
    main()
