import torch
import numpy as np
from cosyvoice.utils.common import IGNORE_ID
import logging

logger = logging.getLogger(__name__)

def xy_data_collator(features, text_tokenizer, xy_tokenizer, num_channels,text_shift_size,speech_vocab_size, device):
    """
    Custom data collator for RWKV7XYLM with MultipleWebDataset.
    Processes a batch of raw data from multiple channels.
    """
    processed_features = []

    # Process each feature
    for feature in features:
        text = f"[S0]{feature.get('json', {}).get('text', '')}[CTL0]"
        audio_np = feature.get('audio', {}).get('array')

        if not text or audio_np is None:
            logger.warning("Skipping sample due to missing text or audio.")
            continue

        text_tokens = text_tokenizer(text, return_tensors="pt").input_ids.squeeze(0)
        
        with torch.no_grad():
            # Convert numpy array to torch tensor and move to device
            audio_tensor = torch.from_numpy(audio_np).to(device)
            encoded_audio = xy_tokenizer.encode([audio_tensor], device=device)
            speech_tokens = encoded_audio['codes_list'][0]
            # Create a new tensor to avoid inplace modification on inference tensor
            speech_tokens = speech_tokens.clone()
            speech_tokens[0,:] = speech_tokens[0,:] + text_shift_size
        
        processed_features.append({'text': text_tokens, 'speech': speech_tokens})

    if not processed_features:
        return {}

    # Constants
    audio_token_pad_token_id = speech_vocab_size-1  # pad id，需与模型 speech_vocab_size 匹配
    text_token_pad_token_id = text_tokenizer.vocab_size-1
    ignore_id = -100

    batch_input_ids = []
    batch_labels = []
    batch_attention_masks = []

    for p_feature in processed_features:
        text_tokens = p_feature['text']  # Shape: [T1]
        speech_tokens = p_feature['speech']  # Shape: [8, T2]
        
        T1 = text_tokens.size(0)
        T2 = speech_tokens.size(1)
        total_steps = T1 + T2 + num_channels - 1
        
        # Initialize input_ids and labels with pad tokens
        input_ids = torch.full((total_steps, num_channels), audio_token_pad_token_id, dtype=torch.long)
        labels = torch.full((total_steps, num_channels), ignore_id, dtype=torch.long)
        
        # Fill text tokens in channel 0
        input_ids[:T1, 0] = text_tokens
        input_ids[T1:, 0] = text_token_pad_token_id
        
        # Fill audio tokens with time shifting
        for t in range(T2 + num_channels - 1):
            step_idx = T1 + t
            for ch in range(num_channels):
                channel_time_shift = ch
                ch_index = t - channel_time_shift
                if ch_index >= 0 and ch_index < T2:
                    input_ids[step_idx, ch] = speech_tokens[ch, ch_index]

        # Generate labels by shifting input_ids
        labels[:-1, :] = input_ids[1:, :].clone()
        
        # Set labels for text part to ignore_id (except the last one)
        labels[:T1-1, :] = ignore_id

        # Set labels for padded values to ignore_id by checking the labels tensor itself
        labels[labels == audio_token_pad_token_id] = ignore_id
        labels[labels == text_token_pad_token_id] = ignore_id
        for i in range(num_channels):
            channel_time_shift = i
            labels[T1+T2-1+channel_time_shift,i] = text_token_pad_token_id if i == 0 else audio_token_pad_token_id

        # Create attention mask (1 for valid tokens, 0 for padding)
        attention_mask = torch.ones(total_steps, dtype=torch.long)
        
        batch_input_ids.append(input_ids)
        batch_labels.append(labels)
        batch_attention_masks.append(attention_mask)

    # Pad all samples to the same length
    max_total_steps = max(input_ids.size(0) for input_ids in batch_input_ids)
    
    padded_input_ids = []
    padded_labels = []
    padded_attention_masks = []
    
    for input_ids, labels, attention_mask in zip(batch_input_ids, batch_labels, batch_attention_masks):
        current_steps = input_ids.size(0)
        pad_steps = max_total_steps - current_steps
        
        if pad_steps > 0:
            # Pad input_ids
            pad_input_ids = torch.full((pad_steps, num_channels), audio_token_pad_token_id, dtype=torch.long)
            pad_input_ids[:, 0] = text_token_pad_token_id
            input_ids = torch.cat([input_ids, pad_input_ids], dim=0)
            
            # Pad labels
            pad_labels = torch.full((pad_steps, num_channels), ignore_id, dtype=torch.long)
            labels = torch.cat([labels, pad_labels], dim=0)
            
            # Pad attention mask
            pad_attention_mask = torch.zeros(pad_steps, dtype=torch.long)
            attention_mask = torch.cat([attention_mask, pad_attention_mask], dim=0)
        
        padded_input_ids.append(input_ids)
        padded_labels.append(labels)
        padded_attention_masks.append(attention_mask)

    # Stack into batch tensors
    final_input_ids = torch.stack(padded_input_ids, dim=0)
    final_labels = torch.stack(padded_labels, dim=0)
    final_attention_mask = torch.stack(padded_attention_masks, dim=0)

    return {
        "input_ids": final_input_ids,
        "labels": final_labels,
        "attention_mask": final_attention_mask
    }

    