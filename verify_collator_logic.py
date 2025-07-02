

import torch
from torch.utils.data import DataLoader
import argparse

# --- Import necessary components from your project ---
from model.llm.xy_llm import RWKV7XYLM
from data.spark.multiple_webdataset import MultipleWebDataset
from XY_Tokenizer.xy_tokenizer.model import XY_Tokenizer
from transformers import AutoTokenizer

# --- Import the two functions to compare ---
# 1. The original collator
from data.utils.collator import xy_data_collator
# 2. The new processing function (we need to add its definition here for a clean test)
#    (Copied from train_scripts/train_xy_llm.py)
import logging
logger = logging.getLogger(__name__)

def process_batch(features, text_tokenizer, xy_tokenizer, num_channels, text_shift_size, speech_vocab_size, device):
    """
    Processes a batch of raw data from multiple channels.
    This function is a standalone copy of the one in train_xy_llm.py for verification.
    """
    processed_features = []
    for feature in features:
        text = f"[SP0]{feature.get('json', {}).get('text', '')}[CTL0]"
        audio_np = feature.get('audio', {}).get('array')
        if not text or audio_np is None:
            logger.warning("Skipping sample due to missing text or audio.")
            continue
        text_tokens = text_tokenizer(text, return_tensors="pt").input_ids.squeeze(0)
        with torch.no_grad():
            audio_tensor = torch.from_numpy(audio_np).to(device)
            encoded_audio = xy_tokenizer.encode([audio_tensor], device=device)
            speech_tokens = encoded_audio['codes_list'][0]
            speech_tokens = speech_tokens.clone()
            speech_tokens[0,:] = speech_tokens[0,:] + text_shift_size
        processed_features.append({'text': text_tokens, 'speech': speech_tokens})

    if not processed_features: return {}
    
    audio_token_pad_token_id = speech_vocab_size-1
    text_token_pad_token_id = text_tokenizer.vocab_size-1
    ignore_id = -100
    batch_input_ids, batch_labels, batch_attention_masks = [], [], []

    for p_feature in processed_features:
        text_tokens, speech_tokens = p_feature['text'], p_feature['speech']
        T1, T2 = text_tokens.size(0), speech_tokens.size(1)
        total_steps = T1 + T2 + num_channels - 1
        input_ids = torch.full((total_steps, num_channels), audio_token_pad_token_id, dtype=torch.long)
        labels = torch.full((total_steps, num_channels), ignore_id, dtype=torch.long)
        input_ids[:T1, 0] = text_tokens
        input_ids[T1:, 0] = text_token_pad_token_id
        for t in range(T2 + num_channels - 1):
            step_idx = T1 + t
            for ch in range(num_channels):
                ch_index = t - ch
                if 0 <= ch_index < T2:
                    input_ids[step_idx, ch] = speech_tokens[ch, ch_index]
        labels[:-1, :] = input_ids[1:, :].clone()
        labels[:T1-1, :] = ignore_id
        labels[labels == audio_token_pad_token_id] = ignore_id
        labels[labels == text_token_pad_token_id] = ignore_id
        for i in range(num_channels):
            labels[T1+T2-1+i, i] = text_token_pad_token_id if i == 0 else audio_token_pad_token_id
        
        attention_mask = torch.ones(total_steps, dtype=torch.long)
        batch_input_ids.append(input_ids)
        batch_labels.append(labels)
        batch_attention_masks.append(attention_mask)

    max_total_steps = max(ids.size(0) for ids in batch_input_ids)
    padded_input_ids, padded_labels, padded_attention_masks = [], [], []

    for ids, labs, masks in zip(batch_input_ids, batch_labels, batch_attention_masks):
        pad_steps = max_total_steps - ids.size(0)
        if pad_steps > 0:
            pad_ids = torch.full((pad_steps, num_channels), audio_token_pad_token_id, dtype=torch.long)
            pad_ids[:, 0] = text_token_pad_token_id
            ids = torch.cat([ids, pad_ids], dim=0)
            pad_labs = torch.full((pad_steps, num_channels), ignore_id, dtype=torch.long)
            labs = torch.cat([labs, pad_labs], dim=0)
            pad_masks = torch.zeros(pad_steps, dtype=torch.long)
            masks = torch.cat([masks, pad_masks], dim=0)
        padded_input_ids.append(ids)
        padded_labels.append(labs)
        padded_attention_masks.append(masks)

    return {
        "input_ids": torch.stack(padded_input_ids, dim=0),
        "labels": torch.stack(padded_labels, dim=0),
        "attention_mask": torch.stack(padded_attention_masks, dim=0)
    }


def compare_outputs(dict1, dict2):
    """Compare two dictionaries of tensors."""
    if dict1.keys() != dict2.keys():
        print("❌ FAILED: Dictionaries have different keys.")
        print(f"  - Keys in output 1: {sorted(dict1.keys())}")
        print(f"  - Keys in output 2: {sorted(dict2.keys())}")
        return False

    all_match = True
    for key in dict1:
        tensor1 = dict1[key]
        tensor2 = dict2[key]
        if not torch.equal(tensor1, tensor2):
            print(f"❌ FAILED: Tensors for key '{key}' do not match.")
            all_match = False
            # Optional: print more details on mismatch
            # print(f"  - Shape mismatch: {tensor1.shape} vs {tensor2.shape}")
            # print(f"  - Dtype mismatch: {tensor1.dtype} vs {tensor2.dtype}")
            # print(f"  - Content mismatch details:\n{tensor1}\nvs\n{tensor2}")
    
    return all_match

def main():
    parser = argparse.ArgumentParser(description="Verify consistency between collator functions.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--xy_tokenizer_config_path", type=str, required=True)
    parser.add_argument("--xy_tokenizer_ckpt_path", type=str, required=True)
    parser.add_argument("--webdataset_dir", type=str, required=True)
    parser.add_argument("--text_shift_size", type=int, default=65536)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load all necessary components ---
    print("1. Loading models and tokenizers...")
    text_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = RWKV7XYLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    xy_tokenizer = XY_Tokenizer.load_from_checkpoint(args.xy_tokenizer_config_path, args.xy_tokenizer_ckpt_path)
    xy_tokenizer.eval().to(device)

    num_channels = model.config.num_channels
    speech_vocab_size = model.config.speech_vocab_size
    
    # --- Load dataset and get a single batch ---
    print("2. Loading dataset and fetching a raw batch...")
    dataset = MultipleWebDataset(data_dir=args.webdataset_dir, target_sr=16000, target_channels=1, shuffle=False) # Shuffle=False for reproducibility
    
    # Use a simple lambda to just get the list of features
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda x: x)
    
    try:
        raw_batch = next(iter(dataloader))
    except StopIteration:
        print("Could not get a batch from the dataloader. Is the dataset empty?")
        return

    print(f"   - Fetched a raw batch of size: {len(raw_batch)}")

    # --- Run both functions ---
    print("3. Running both processing functions...")
    
    # Run the original xy_data_collator
    output_original = xy_data_collator(
        raw_batch, text_tokenizer, xy_tokenizer, num_channels, 
        args.text_shift_size, speech_vocab_size, device
    )
    print("   - Original `xy_data_collator` executed.")

    # Run the new process_batch function
    output_new = process_batch(
        raw_batch, text_tokenizer, xy_tokenizer, num_channels, 
        args.text_shift_size, speech_vocab_size, device
    )
    print("   - New `process_batch` executed.")

    # --- Compare the results ---
    print("4. Comparing outputs...")
    if compare_outputs(output_original, output_new):
        print("\n✅ SUCCESS: The outputs of `xy_data_collator` and `process_batch` are identical.")
    else:
        print("\n❌ FAILURE: The outputs do not match. Please review the differences printed above.")

if __name__ == "__main__":
    main()

