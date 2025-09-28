import torch
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class XYDataProcessor:
    """
    A utility class to process raw batches from the JSONL dataset into a format
    suitable for the RWKV7XYLM model.
    """
    def __init__(self, text_tokenizer, num_channels, text_shift_size, speech_vocab_size):
        """
        Initializes the data processor.

        Args:
            text_tokenizer: The tokenizer for processing text.
            num_channels (int): The number of audio channels in the model.
            text_shift_size (int): The value to shift the first audio channel tokens by.
            speech_vocab_size (int): The vocabulary size for speech tokens.
        """
        self.text_tokenizer = text_tokenizer
        self.num_channels = num_channels
        self.text_shift_size = text_shift_size
        self.speech_vocab_size = speech_vocab_size
        self.audio_token_pad_token_id = speech_vocab_size - 1
        self.text_token_pad_token_id = text_tokenizer.vocab_size - 1
        self.ignore_id = -100

    def process_batch(self, raw_batch: Dict[str, List[Any]]) -> Dict[str, torch.Tensor]:
        """
        Processes a raw batch of data.

        Args:
            raw_batch: A dictionary containing lists of 'text' and 'audio_tokens'.

        Returns:
            A dictionary containing the final 'input_ids', 'labels', and 'attention_mask' tensors.
        """
        processed_features = []

        # Iterate over each sample in the batch
        for i in range(len(raw_batch['text'])):
            text = f"[S0]{raw_batch['text'][i]}[CTL0]"
            speech_tokens_list = raw_batch['audio_tokens'][i]

            if not text or speech_tokens_list is None:
                logger.warning("Skipping sample due to missing text or audio_tokens.")
                continue

            text_tokens = self.text_tokenizer(text, return_tensors="pt").input_ids.squeeze(0)
            speech_tokens = torch.tensor(speech_tokens_list, dtype=torch.long)
            
            # Apply the token shift to the first audio channel
            speech_tokens = speech_tokens.clone()
            speech_tokens[0, :] = speech_tokens[0, :] + self.text_shift_size

            processed_features.append({'text': text_tokens, 'speech': speech_tokens})

        if not processed_features:
            return {}

        batch_input_ids = []
        batch_labels = []
        batch_attention_masks = []

        # Construct input_ids, labels, and attention_mask for each sample
        for p_feature in processed_features:
            text_tokens = p_feature['text']
            speech_tokens = p_feature['speech']

            T1 = text_tokens.size(0)
            T2 = speech_tokens.size(1)
            total_steps = T1 + T2 + self.num_channels - 1

            input_ids = torch.full((total_steps, self.num_channels), self.audio_token_pad_token_id, dtype=torch.long)
            labels = torch.full((total_steps, self.num_channels), self.ignore_id, dtype=torch.long)

            # Fill in text and audio tokens
            input_ids[:T1, 0] = text_tokens
            input_ids[T1:, 0] = self.text_token_pad_token_id

            for t in range(T2 + self.num_channels - 1):
                step_idx = T1 + t
                for ch in range(self.num_channels):
                    ch_index = t - ch
                    if 0 <= ch_index < T2:
                        input_ids[step_idx, ch] = speech_tokens[ch, ch_index]

            # Generate labels by shifting inputs
            labels[:-1, :] = input_ids[1:, :].clone()
            labels[:T1-1, :] = self.ignore_id
            labels[labels == self.audio_token_pad_token_id] = self.ignore_id
            labels[labels == self.text_token_pad_token_id] = self.ignore_id
            for i in range(self.num_channels):
                labels[T1 + T2 - 1 + i, i] = self.text_token_pad_token_id if i == 0 else self.audio_token_pad_token_id

            attention_mask = torch.ones(total_steps, dtype=torch.long)

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_masks.append(attention_mask)

        # Pad all samples to the same length
        max_total_steps = max(ids.size(0) for ids in batch_input_ids)

        padded_input_ids, padded_labels, padded_attention_masks = [], [], []

        for input_ids, labels, attention_mask in zip(batch_input_ids, batch_labels, batch_attention_masks):
            pad_steps = max_total_steps - input_ids.size(0)
            if pad_steps > 0:
                pad_input_ids = torch.full((pad_steps, self.num_channels), self.audio_token_pad_token_id, dtype=torch.long)
                pad_input_ids[:, 0] = self.text_token_pad_token_id
                input_ids = torch.cat([input_ids, pad_input_ids], dim=0)

                pad_labels = torch.full((pad_steps, self.num_channels), self.ignore_id, dtype=torch.long)
                labels = torch.cat([labels, pad_labels], dim=0)

                pad_attention_mask = torch.zeros(pad_steps, dtype=torch.long)
                attention_mask = torch.cat([attention_mask, pad_attention_mask], dim=0)

            padded_input_ids.append(input_ids)
            padded_labels.append(labels)
            padded_attention_masks.append(attention_mask)

        return {
            "input_ids": torch.stack(padded_input_ids, dim=0),
            "labels": torch.stack(padded_labels, dim=0),
            "attention_mask": torch.stack(padded_attention_masks, dim=0)
        }
