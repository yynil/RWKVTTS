import torch
from utils.properties_util import convert_properties_to_tokens

def create_inputs_and_labels(batch, tokenizer, model, eos_token_id, device):
    texts = batch['text']
    global_tokens_list = batch['global_tokens']
    semantic_tokens_list = batch['semantic_tokens']

    all_input_embs_list = []
    all_labels_list = []

    for i in range(len(texts)):
        # 1. Get token IDs
        text_tokens = tokenizer.encode(texts[i], add_special_tokens=False)
        global_tokens = global_tokens_list[i]
        semantic_tokens = semantic_tokens_list[i]

        # 2. Convert to tensors on device
        text_tokens_tensor = torch.tensor(text_tokens, dtype=torch.long, device=device)
        global_tokens_tensor = torch.tensor(global_tokens, dtype=torch.long, device=device)
        semantic_tokens_to_predict = torch.tensor(semantic_tokens + [eos_token_id], dtype=torch.long, device=device)

        # 3. Get embeddings
        text_embs = model.text_embedder(text_tokens_tensor)
        global_embs = model.global_embedder(global_tokens_tensor)
        semantic_embs = model.model.embeddings(semantic_tokens_to_predict)

        # 4. Get special tag embeddings
        tag_0_emb = model.tts_tag_embedder(torch.tensor([0], dtype=torch.long, device=device))
        tag_1_emb = model.tts_tag_embedder(torch.tensor([1], dtype=torch.long, device=device))
        tag_2_emb = model.tts_tag_embedder(torch.tensor([2], dtype=torch.long, device=device))

        # 5. Concatenate embeddings for one sample
        full_embs_for_sample = torch.cat([
            tag_2_emb, 
            text_embs, 
            tag_0_emb, 
            global_embs, 
            tag_1_emb, 
            semantic_embs
        ], dim=0)
        all_input_embs_list.append(full_embs_for_sample)

        # 6. Create labels for one sample, aligned with the input embeddings.
        # The model's forward pass will handle shifting. We only calculate loss on semantic tokens.
        prefix_len = 1 + len(text_tokens) + 1 + len(global_tokens) + 1 # TAG2, TEXT, TAG0, GLOBAL, TAG1
        prefix_labels = torch.full((prefix_len,), -100, dtype=torch.long, device=device)
        
        # Labels are aligned with inputs. Lengths must match.
        labels_for_sample = torch.cat([
            prefix_labels,
            semantic_tokens_to_predict
        ], dim=0)
        all_labels_list.append(labels_for_sample)

    # Pad the lists of tensors
    padded_input_embs = torch.nn.utils.rnn.pad_sequence(
        all_input_embs_list, batch_first=True, padding_value=0.0
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        all_labels_list, batch_first=True, padding_value=-100
    )

    # Create attention mask
    lengths = [len(s) for s in all_input_embs_list]
    attention_mask = torch.zeros(len(texts), max(lengths), device=device, dtype=torch.long)
    for i, l in enumerate(lengths):
        attention_mask[i, :l] = 1

    return {
        "input_embs": padded_input_embs,
        "labels": padded_labels,
        "attention_mask": attention_mask
    }

def create_inputs_and_labels_culens(batch, tokenizer, model, eos_token_id, device):
    texts = batch['text']
    global_tokens_list = batch['global_tokens']
    semantic_tokens_list = batch['semantic_tokens']
    input_ids_embs_list = []
    labels_list = []
    cu_seqlens = [0]

    for i in range(len(texts)):
        # 1. Get token IDs
        text_tokens = tokenizer.encode(texts[i], add_special_tokens=False)
        global_tokens = global_tokens_list[i]
        semantic_tokens = semantic_tokens_list[i]

        # 2. Convert to tensors on device
        text_tokens_tensor = torch.tensor(text_tokens, dtype=torch.long, device=device)
        global_tokens_tensor = torch.tensor(global_tokens, dtype=torch.long, device=device)
        semantic_tokens_to_predict = torch.tensor(semantic_tokens + [eos_token_id], dtype=torch.long, device=device)

        # 3. Get embeddings
        text_embs = model.text_embedder(text_tokens_tensor)
        global_embs = model.global_embedder(global_tokens_tensor)
        semantic_embs = model.model.embeddings(semantic_tokens_to_predict)

        # 4. Get special tag embeddings
        tag_0_emb = model.tts_tag_embedder(torch.tensor([0], dtype=torch.long, device=device))
        tag_1_emb = model.tts_tag_embedder(torch.tensor([1], dtype=torch.long, device=device))
        tag_2_emb = model.tts_tag_embedder(torch.tensor([2], dtype=torch.long, device=device))

        # 5. Concatenate embeddings for one sample
        full_embs_for_sample = torch.cat([
            tag_2_emb, 
            text_embs, 
            tag_0_emb, 
            global_embs, 
            tag_1_emb, 
            semantic_embs
        ], dim=0)
        input_ids_embs_list.append(full_embs_for_sample)

        # 6. Create labels for one sample, aligned with the input embeddings.
        prefix_len = 1 + len(text_tokens) + 1 + len(global_tokens) + 1 # TAG2, TEXT, TAG0, GLOBAL, TAG1
        prefix_labels = torch.full((prefix_len,), -100, dtype=torch.long, device=device)
        
        labels_for_sample = torch.cat([
            prefix_labels,
            semantic_tokens_to_predict
        ], dim=0)
        labels_list.append(labels_for_sample)
        
        cu_seqlens.append(cu_seqlens[-1] + len(full_embs_for_sample))
    
    input_embs = torch.cat(input_ids_embs_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.long, device=device)
    
    return {
        "input_embs": input_embs.unsqueeze(0),
        "labels": labels.unsqueeze(0),
        "cu_seqlens": cu_seqlens
    }

global_debug = True
def create_inputs_and_labels_with_properties(batch, tokenizer, model, eos_token_id, device):
    global global_debug
    texts = batch['text']
    global_tokens_list = batch['global_tokens']
    semantic_tokens_list = batch['semantic_tokens']
    ages = batch['age']
    genders = batch['gender']
    emotions = batch['emotion']
    pitches = batch['pitch']
    speeds = batch['speed']


    all_input_embs_list = []
    all_labels_list = []

    for i in range(len(texts)):
        # --- Common setup for both samples ---
        text = texts[i]
        text_tokens = tokenizer.encode(text, add_special_tokens=False)
        global_tokens = global_tokens_list[i]
        semantic_tokens = semantic_tokens_list[i]

        text_tokens_tensor = torch.tensor(text_tokens, dtype=torch.long, device=device)
        global_tokens_tensor = torch.tensor(global_tokens, dtype=torch.long, device=device)
        semantic_tokens_to_predict = torch.tensor(semantic_tokens + [eos_token_id], dtype=torch.long, device=device)

        text_embs = model.text_embedder(text_tokens_tensor)
        global_embs = model.global_embedder(global_tokens_tensor)
        semantic_embs = model.model.embeddings(semantic_tokens_to_predict)

        tag_0_emb = model.tts_tag_embedder(torch.tensor([0], dtype=torch.long, device=device))
        tag_1_emb = model.tts_tag_embedder(torch.tensor([1], dtype=torch.long, device=device))
        tag_2_emb = model.tts_tag_embedder(torch.tensor([2], dtype=torch.long, device=device))

        # --- Sample 1: Standard TTS (no properties) ---
        full_embs_for_sample = torch.cat([
            tag_2_emb, text_embs, tag_0_emb, global_embs, tag_1_emb, semantic_embs
        ], dim=0)
        
        prefix_len_no_props = 1 + len(text_tokens) + 1 + len(global_tokens) + 1
        prefix_labels_no_props = torch.full((prefix_len_no_props,), -100, dtype=torch.long, device=device)
        labels_for_sample = torch.cat([prefix_labels_no_props, semantic_tokens_to_predict], dim=0)

        all_input_embs_list.append(full_embs_for_sample)
        all_labels_list.append(labels_for_sample)

        # --- Sample 2: Controllable TTS (with properties) ---
        properties_str = convert_properties_to_tokens(ages[i], genders[i], emotions[i], pitches[i], speeds[i])
        properties_tokens = tokenizer.encode(properties_str, add_special_tokens=False)
        if global_debug:
            print(f"properties_str: {properties_str}")
            print(f"properties_tokens: {properties_tokens}")
            global_debug = False
        properties_tokens_tensor = torch.tensor(properties_tokens, dtype=torch.long, device=device)
        properties_embs = model.text_embedder(properties_tokens_tensor)

        full_embs_with_properties = torch.cat([
            properties_embs, full_embs_for_sample
        ], dim=0)

        # Goal: Predict global and semantic tokens
        prefix_len_props = len(properties_tokens) + 1 + len(text_tokens) + 1 # PROPS, TAG2, TEXT, TAG0
        prefix_labels_props = torch.full((prefix_len_props,), -100, dtype=torch.long, device=device)
        ignore_tag1_label = torch.full((1,), -100, dtype=torch.long, device=device) # for TAG1

        # Aligned labels: [IGNORE(PROPS..TAG0), GLOBAL_TOKENS, IGNORE(TAG1), SEMANTIC_TOKENS]
        labels_for_sample_with_properties = torch.cat([
            prefix_labels_props,
            global_tokens_tensor,
            ignore_tag1_label,
            semantic_tokens_to_predict
        ], dim=0)
        
        all_input_embs_list.append(full_embs_with_properties)
        all_labels_list.append(labels_for_sample_with_properties)

    # Pad the lists of tensors
    padded_input_embs = torch.nn.utils.rnn.pad_sequence(
        all_input_embs_list, batch_first=True, padding_value=0.0
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        all_labels_list, batch_first=True, padding_value=-100
    )

    # Create attention mask
    lengths = [len(s) for s in all_input_embs_list]
    attention_mask = torch.zeros(len(texts)*2, max(lengths), device=device, dtype=torch.long)
    for i, l in enumerate(lengths):
        attention_mask[i, :l] = 1

    return {
        "input_embs": padded_input_embs,
        "labels": padded_labels,
        "attention_mask": attention_mask
    }


def create_inputs_and_labels_with_properties_culens(batch, tokenizer, model, eos_token_id, device):
    texts = batch['text']
    global_tokens_list = batch['global_tokens']
    semantic_tokens_list = batch['semantic_tokens']
    ages = batch['age']
    genders = batch['gender']
    emotions = batch['emotion']
    pitches = batch['pitch']
    speeds = batch['speed']
    input_ids_embs_list = []
    labels_list = []
    cu_seqlens = [0]

    for i in range(len(texts)):
        # --- Common setup for both samples ---
        text = texts[i]
        text_tokens = tokenizer.encode(text, add_special_tokens=False)
        global_tokens = global_tokens_list[i]
        semantic_tokens = semantic_tokens_list[i]

        text_tokens_tensor = torch.tensor(text_tokens, dtype=torch.long, device=device)
        global_tokens_tensor = torch.tensor(global_tokens, dtype=torch.long, device=device)
        semantic_tokens_to_predict = torch.tensor(semantic_tokens + [eos_token_id], dtype=torch.long, device=device)

        text_embs = model.text_embedder(text_tokens_tensor)
        global_embs = model.global_embedder(global_tokens_tensor)
        semantic_embs = model.model.embeddings(semantic_tokens_to_predict)

        tag_0_emb = model.tts_tag_embedder(torch.tensor([0], dtype=torch.long, device=device))
        tag_1_emb = model.tts_tag_embedder(torch.tensor([1], dtype=torch.long, device=device))
        tag_2_emb = model.tts_tag_embedder(torch.tensor([2], dtype=torch.long, device=device))

        # --- Sample 1: Standard TTS (no properties) ---
        full_embs_for_sample = torch.cat([
            tag_2_emb, text_embs, tag_0_emb, global_embs, tag_1_emb, semantic_embs
        ], dim=0)
        
        prefix_len_no_props = 1 + len(text_tokens) + 1 + len(global_tokens) + 1
        prefix_labels_no_props = torch.full((prefix_len_no_props,), -100, dtype=torch.long, device=device)
        labels_for_sample = torch.cat([prefix_labels_no_props, semantic_tokens_to_predict], dim=0)

        input_ids_embs_list.append(full_embs_for_sample)
        labels_list.append(labels_for_sample)
        cu_seqlens.append(cu_seqlens[-1] + len(full_embs_for_sample))

        # --- Sample 2: Controllable TTS (with properties) ---
        properties_str = convert_properties_to_tokens(ages[i], genders[i], emotions[i], pitches[i], speeds[i])
        properties_tokens = tokenizer.encode(properties_str, add_special_tokens=False)
        properties_tokens_tensor = torch.tensor(properties_tokens, dtype=torch.long, device=device)
        properties_embs = model.text_embedder(properties_tokens_tensor)

        full_embs_with_properties = torch.cat([
            properties_embs, full_embs_for_sample
        ], dim=0)

        prefix_len_props = len(properties_tokens) + 1 + len(text_tokens) + 1
        prefix_labels_props = torch.full((prefix_len_props,), -100, dtype=torch.long, device=device)
        ignore_tag1_label = torch.full((1,), -100, dtype=torch.long, device=device)

        labels_for_sample_with_properties = torch.cat([
            prefix_labels_props, global_tokens_tensor, ignore_tag1_label, semantic_tokens_to_predict
        ], dim=0)
        
        input_ids_embs_list.append(full_embs_with_properties)
        labels_list.append(labels_for_sample_with_properties)
        cu_seqlens.append(cu_seqlens[-1] + len(full_embs_with_properties))

    input_embs = torch.cat(input_ids_embs_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.long, device=device)
    
    return {
        "input_embs": input_embs.unsqueeze(0),
        "labels": labels.unsqueeze(0),
        "cu_seqlens": cu_seqlens
    }
