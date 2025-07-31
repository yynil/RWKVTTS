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
        semantic_tokens_tensor = torch.tensor(semantic_tokens + [eos_token_id], dtype=torch.long, device=device)

        # 3. Get embeddings
        text_embs = model.text_embedder(text_tokens_tensor)
        global_embs = model.global_embedder(global_tokens_tensor)
        semantic_embs = model.model.embeddings(semantic_tokens_tensor)

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

        # 6. Create labels for one sample
        prefix_len = 1 + len(text_tokens) + 1 + len(global_tokens)
        labels_for_sample = torch.cat([
            torch.full((prefix_len,), -100, dtype=torch.long, device=device),
            semantic_tokens_tensor,
            torch.tensor([-100], dtype=torch.long, device=device)
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
        age = ages[i]
        gender = genders[i]
        emotion = emotions[i]
        pitch = pitches[i]
        speed = speeds[i]
        properties_tokens = convert_properties_to_tokens(age, gender, emotion, pitch, speed)
        if global_debug:
            print(f"Befor tokenization properties_tokens: {properties_tokens}")
        text = texts[i]
        properties_tokens = tokenizer.encode(properties_tokens, add_special_tokens=False)
        if global_debug:
            print(f"After tokenization properties_tokens: {properties_tokens}")
            global_debug = False
        # 1. Get token IDs
        text_tokens = tokenizer.encode(text, add_special_tokens=False)
        global_tokens = global_tokens_list[i]
        semantic_tokens = semantic_tokens_list[i]

        # 2. Convert to tensors on device
        text_tokens_tensor = torch.tensor(text_tokens, dtype=torch.long, device=device)
        properties_tokens_tensor = torch.tensor(properties_tokens, dtype=torch.long, device=device)
        global_tokens_tensor = torch.tensor(global_tokens, dtype=torch.long, device=device)
        semantic_tokens_tensor = torch.tensor(semantic_tokens + [eos_token_id], dtype=torch.long, device=device)

        # 3. Get embeddings
        text_embs = model.text_embedder(text_tokens_tensor)
        properties_embs = model.text_embedder(properties_tokens_tensor)
        global_embs = model.global_embedder(global_tokens_tensor)
        semantic_embs = model.model.embeddings(semantic_tokens_tensor)

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
        full_embs_with_properties = torch.cat([
            properties_embs,
            full_embs_for_sample
        ], dim=0)
        all_input_embs_list.append(full_embs_for_sample)
        all_input_embs_list.append(full_embs_with_properties)

        # 6. Create labels for one sample
        prefix_len = 1 + len(text_tokens) + 1 + len(global_tokens)
        labels_for_sample = torch.cat([
            torch.full((prefix_len,), -100, dtype=torch.long, device=device),
            semantic_tokens_tensor,
            torch.tensor([-100], dtype=torch.long, device=device)
        ], dim=0)
        #create labels for one sample with properties which makes global tokens and semantic tokens as training target
        prefix_len = len(properties_tokens) + 1 + len(text_tokens)
        labels_for_sample_with_properties = torch.cat([
            torch.full((prefix_len,), -100, dtype=torch.long, device=device),
            global_tokens_tensor,
            torch.tensor([-100], dtype=torch.long, device=device),
            semantic_tokens_tensor,
            torch.tensor([-100], dtype=torch.long, device=device)
        ], dim=0)
        all_labels_list.append(labels_for_sample)
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
