import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
import deepspeed
from transformers import HfArgumentParser, AutoTokenizer
from dataclasses import dataclass, field
import logging
import json
from typing import Optional, List, Dict
from functools import partial
import time
from torch.nn.utils.rnn import pad_sequence
import wandb
import soundfile as sf
import numpy as np
import io
import glob
import random
import math

# Custom model and data components
from model.llm.xy_llm import RWKV7XYLM
from data.spark.multiple_webdataset import MultipleWebDataset
from cosyvoice.utils.common import IGNORE_ID
from XY_Tokenizer.xy_tokenizer.model import XY_Tokenizer

logger = logging.getLogger(__name__)

@dataclass
class ScriptArguments:
    """Command line arguments for training the RWKV7XYLM model"""
    model_name_or_path: str = field(
        metadata={"help": "Path to the converted RWKV7XYLM model directory."}
    )
    xy_tokenizer_config_path: str = field(
        metadata={"help": "Path to the XY_Tokenizer config YAML file."}
    )
    xy_tokenizer_ckpt_path: str = field(
        metadata={"help": "Path to the XY_Tokenizer checkpoint file."}
    )
    webdataset_dir: str = field(
        metadata={"help": "Path to the webdataset directory containing training data."}
    )
    output_dir: str = field(
        metadata={"help": "Directory to save the trained model and checkpoints."}
    )
    max_tokens_per_round: Optional[int] = field(
        default=None, 
        metadata={"help": "Maximum number of tokens (B x T) per round in KB (1024s). If a batch exceeds this, it will be sliced. E.g., 16 means 16384 tokens."}
    )
    # --- Deepspeed and Training Hyperparameters ---
    deepspeed_config: Optional[str] = field(
        default=None, metadata={"help": "Path to DeepSpeed config file."}
    )
    num_epochs: int = field(default=3, metadata={"help": "Number of training epochs."})
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Training batch size per device."}
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "Initial learning rate."})
    learning_rate_final: float = field(default=1e-6, metadata={"help": "Final learning rate."})
    warmup_steps: int = field(default=200, metadata={"help": "Number of warmup steps."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay."})
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "Enable gradient checkpointing."}
    )
    # --- DeepSpeed specific ---
    ds_stage: int = field(default=3, metadata={"help": "DeepSpeed stage"})
    ds_param_offload: bool = field(default=True, metadata={"help": "DeepSpeed parameter offload"})
    ds_optimizer_offload: bool = field(default=True, metadata={"help": "DeepSpeed optimizer offload"})
    # --- Logging and Saving ---
    logging_steps: int = field(default=20, metadata={"help": "Log every N steps."})
    save_steps: int = field(default=500, metadata={"help": "Save a checkpoint every N steps."})
    wandb_project: str = field(
        default="rwkv7-xy-lm-training", metadata={"help": "Name of W&B project."}
    )
    wandb_run_name: Optional[str] = field(
        default=None, metadata={"help": "Name of W&B run."}
    )
    # --- System and Environment ---
    local_rank: int = field(default=-1, metadata={"help": "Local rank for distributed training."})
    seed: int = field(default=42, metadata={"help": "Random seed."})

def simple_collate_fn(features):
    """A simple collate function that just returns the features."""
    return features

def process_batch(features, text_tokenizer, xy_tokenizer, num_channels, text_shift_size, speech_vocab_size, device):
    """
    Processes a batch of raw data from multiple channels.
    This function is designed to be called within the training loop,
    after the data has been loaded by the DataLoader.
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

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def configure_optimizer(model, args):
    """Configure optimizer with parameter grouping"""
    lr_1x = set()
    lr_2x = set()
    lr_decay = set()    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'attn.w_lora.lora.2.bias' in n:
            lr_2x.add(n)
        elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0) and (".weight" in n) and ("lora" not in n):
            lr_decay.add(n)
        else:
            lr_1x.add(n)

    lr_1x = sorted(list(lr_1x))
    lr_2x = sorted(list(lr_2x))
    lr_decay = sorted(list(lr_decay))
    param_dict = {n: p for n, p in model.named_parameters()}
    
    optim_groups = [
        {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0, "name": "lr_1x"},
        {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0, "name": "lr_2x"}
    ]
    if args.weight_decay > 0:
        optim_groups.append({
            "params": [param_dict[n] for n in lr_decay],
            "weight_decay": args.weight_decay,
            "my_lr_scale": 1.0,
            "name": "lr_decay"
        })
        adamw_mode = True
    else:
        adamw_mode = False
    
    if args.ds_optimizer_offload:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        optimizer = DeepSpeedCPUAdam(optim_groups, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-18, bias_correction=True, adamw_mode=adamw_mode, amsgrad=False, weight_decay=args.weight_decay)
    else:
        from deepspeed.ops.adam import FusedAdam
        optimizer = FusedAdam(optim_groups, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-18, bias_correction=True, adam_w_mode=adamw_mode, amsgrad=False, weight_decay=args.weight_decay)
  
    return optimizer

def update_learning_rate(optimizer, current_step, total_steps, warmup_steps, learning_rate, learning_rate_final, args, is_main_process):
    """更新优化器中每个参数组的学习率"""
    # 计算基础学习率
    if current_step < warmup_steps:
        base_lr = learning_rate * (0.01 + 0.99 * current_step / warmup_steps)
    else:
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = max(0, min(1, progress))
        lr_final_factor = learning_rate_final / learning_rate
        base_lr = learning_rate * ((0.5 + lr_final_factor / 2) + (0.5 - lr_final_factor / 2) * math.cos(math.pi * progress))
    
    # 更新每个参数组的学习率
    for param_group in optimizer.param_groups:
        if param_group.get('weight_decay', 0) > 0:
            param_group['weight_decay'] = args.weight_decay
        lr_scale = param_group.get('my_lr_scale', 1.0)
        param_group['lr'] = base_lr * lr_scale
        if is_main_process and current_step % 100 == 0:
            print(f'param_group: {param_group["name"]} lr: {param_group["lr"]} weight_decay: {param_group["weight_decay"]}')

def log_metrics(optimizer, loss, avg_loss, epoch, total_steps, kts, all_tokens, current_lr):
    """记录训练指标到 wandb"""
    # 记录基本训练指标
    wandb.log({
        "loss": loss.item(),
        "avg_loss": avg_loss,
        "epoch": epoch,
        "step": total_steps,
        "KT/s": kts,
        "Gtokens": all_tokens/1e9,
        "learning_rate": current_lr
    })
    
    # 记录每个参数组的学习率和权重衰减
    for param_group in optimizer.param_groups:
        # 计算参数组的统计信息
        params = param_group['params']
        total_params = sum(p.numel() for p in params)
        
        # 记录到 wandb
        wandb.log({
            f"lr_group_{param_group.get('name', 'default')}": param_group['lr'],
            f"wd_group_{param_group.get('name', 'default')}": param_group.get('weight_decay', 0),
            f"params_count_{param_group.get('name', 'default')}": total_params
        })

def save_checkpoint(model_engine, output_dir, epoch, step, logger):
    """Save model checkpoint"""
    if os.path.exists(output_dir):
        if model_engine.local_rank == 0:
            checkpoints = os.listdir(output_dir)
            # only list the directories   
            checkpoints = [f for f in checkpoints if os.path.isdir(os.path.join(output_dir, f))]
            # sort by creation time  
            checkpoints.sort(key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
            if len(checkpoints) > 2:
                print(f'deleting older checkpoints {checkpoints[0]}')
                import shutil
                shutil.rmtree(os.path.join(output_dir, checkpoints[0]))    
    output_dir = f"{output_dir}/epoch_{epoch}_step_{step}"
    print(f'saving checkpoint to {output_dir}')
    if model_engine.local_rank == 0 and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_engine.save_checkpoint(output_dir)

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    is_main_process = local_rank == 0
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()

    if is_main_process:
        logging.basicConfig(level=logging.INFO)
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    logger.info(f"Starting training with arguments: {args}")

    # Load text tokenizer
    text_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    # Load model
    model = RWKV7XYLM.from_pretrained(
        args.model_name_or_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    )
    model.zero_embs()
    logger.info("Padding embeddings have been zeroed out.")
    num_channels = model.config.num_channels
    logger.info(f"Model configured with {num_channels} channels.")

    # Load XY_Tokenizer
    logger.info(f"Loading XY_Tokenizer from config: {args.xy_tokenizer_config_path} and checkpoint: {args.xy_tokenizer_ckpt_path}")
    xy_tokenizer = XY_Tokenizer.load_from_checkpoint(args.xy_tokenizer_config_path, args.xy_tokenizer_ckpt_path)
    xy_tokenizer.eval().to(torch.device(f'cuda:{local_rank}'))
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Prepare dataset
    logger.info(f"Loading dataset from directory: {args.webdataset_dir}")
    dataset = MultipleWebDataset(
        data_dir=args.webdataset_dir,
        target_sr=16000,
        target_channels=1,
        shuffle=True,
        verify_tar=False
    )
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    
    # Create data collator with device parameter
    data_collator = simple_collate_fn
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.per_device_train_batch_size, 
        sampler=sampler,
        num_workers=4, 
        pin_memory=True, 
        collate_fn=data_collator, 
        persistent_workers=True
    )

    # Configure optimizer with parameter grouping
    logger.info("Configuring optimizer with parameter grouping...")
    optimizer = configure_optimizer(model, args)
    
    # Load DeepSpeed config
    if args.deepspeed_config:
        if is_main_process:
            logger.info(f"Loading DeepSpeed config from {args.deepspeed_config}")
        with open(args.deepspeed_config, 'r') as f:
            ds_config = json.load(f)
    else:
        # Default DeepSpeed config
        if is_main_process:
            logger.info("Using default DeepSpeed config")
        train_batch_size = args.per_device_train_batch_size * world_size
        ds_config = {
            "distributed_backend": "nccl",
            "train_batch_size": train_batch_size,
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": args.ds_stage,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_prefetch_bucket_size": 5e6,
                "memory_efficient_linear": True,
                "stage3_param_persistence_threshold": 1e4,
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True,
                    "buffer_count": 4,
                    "buffer_size": 1e8
                },
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True,
                    "buffer_count": 4
                },
                "allgather_partitions": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e6,
                "overlap_comm": False,
                "contiguous_gradients": True
            },
            "zero_force_ds_cpu_initialization": True,
            "gradient_checkpointing": args.gradient_checkpointing,
            "dump_state": False
        }
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config=ds_config
    )

    # Calculate total training steps
    total_steps = len(dataloader) * args.num_epochs
    
    # Create output directory if it doesn't exist
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    logger.info("*** Starting Training ***")
    args.text_shift_size = text_tokenizer.encode("[SP0]")[0]
    print(f"text_shift_size: {args.text_shift_size}")
    # Test a single batch to ensure everything works
    logger.info("Testing data pipeline with a single batch...")
    raw_test_batch = next(iter(dataloader))
    if raw_test_batch:
        test_batch = process_batch(
            raw_test_batch,
            text_tokenizer,
            xy_tokenizer,
            num_channels,
            args.text_shift_size,
            model.config.speech_vocab_size,
            model_engine.device
        )
        if test_batch:
            logger.info(f"Test batch shapes - input_ids: {test_batch['input_ids'].shape}, labels: {test_batch['labels'].shape}, attention_mask: {test_batch['attention_mask'].shape}")
            logger.info("Data pipeline test successful!")
        else:
            logger.warning("Test batch is empty after processing, please check your data!")
    else:
        logger.warning("Test batch is empty, please check your data!")
    
    global_step = 0
    total_loss = 0.0
    all_tokens = 0
    
    try:
        for epoch in range(args.num_epochs):
            model_engine.train()
            sampler.set_epoch(epoch)
            
            if is_main_process:
                update_time = time.time()
                tokens_since_last_log = 0
                logger.info(f"Epoch {epoch} starts training")
                from tqdm import tqdm
                pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")
            
            for step, raw_batch in enumerate(dataloader):
                if not raw_batch: 
                    logger.warning("Empty batch received, skipping...")
                    continue

                # Process the batch
                batch = process_batch(
                    raw_batch,
                    text_tokenizer,
                    xy_tokenizer,
                    num_channels,
                    args.text_shift_size,
                    model.config.speech_vocab_size,
                    model_engine.device
                )

                if not batch: 
                    logger.warning("Empty batch after processing, skipping...")
                    continue

                # --- Dynamic Batch Slicing for Memory Safety ---
                if args.max_tokens_per_round:
                    B, T, C = batch['input_ids'].shape
                    total_tokens = B * T
                    max_tokens = args.max_tokens_per_round * 1024
                    if total_tokens > max_tokens:
                        max_bsz = max_tokens // T
                        if max_bsz > 0:
                            logger.warning(
                                f"Batch token size ({total_tokens}) > max ({max_tokens}). "
                                f"Slicing batch from {B} to {max_bsz}."
                            )
                            batch['input_ids'] = batch['input_ids'][:max_bsz]
                            batch['labels'] = batch['labels'][:max_bsz]
                            batch['attention_mask'] = batch['attention_mask'][:max_bsz]
                        else:
                            logger.warning(
                                f"A single sample with sequence length {T} exceeds max_tokens_per_round. "
                                f"Skipping batch."
                            )
                            continue

                # Update learning rate
                update_learning_rate(
                    optimizer,
                    global_step,
                    total_steps,
                    args.warmup_steps,
                    args.learning_rate,
                    args.learning_rate_final,
                    args,
                    is_main_process
                )

                try:
                    input_ids = batch['input_ids'].to(model_engine.device)
                    labels = batch['labels'].to(model_engine.device)
                    attention_mask = batch.get('attention_mask').to(model_engine.device)

                    outputs = model_engine(input_ids=input_ids, labels=labels, attention_mask=attention_mask, return_dict=True, use_cache=False)
                    loss = outputs.loss

                    model_engine.backward(loss)
                    model_engine.step()

                    global_step += 1

                    # Logging and metrics
                    if is_main_process:
                        total_loss += loss.item()
                        
                        # Accumulate tokens for KT/s calculation
                        batch_tokens = input_ids.numel() # B * T * C
                        tokens_since_last_log += batch_tokens * world_size
                        all_tokens += batch_tokens * world_size

                        if global_step % args.logging_steps == 0:
                            elapsed_time = time.time() - update_time
                            # Prevent division by zero
                            kts = (tokens_since_last_log / elapsed_time / 1e3) if elapsed_time > 0 else 0.0
                            avg_loss = total_loss / args.logging_steps
                            current_lr = optimizer.param_groups[0]['lr']
                            
                            logger.info(
                                f"Epoch: {epoch}, Step: {global_step}, Loss: {loss.item():.4f}, "
                                f"Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}, KT/s: {kts:.2f}"
                            )
                            log_metrics(optimizer, loss, avg_loss, epoch, global_step, kts, all_tokens, current_lr)
                            
                            # Reset for next logging interval
                            total_loss = 0.0
                            tokens_since_last_log = 0
                            update_time = time.time()
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            'loss': loss.item(),
                            'avg_loss': total_loss / (step % args.logging_steps + 1) if args.logging_steps > 0 else total_loss,
                            'lr': optimizer.param_groups[0]['lr']
                        })

                    # --- Padding Embedding Verification ---
                    if is_main_process and global_step % 100 == 0:
                        try:
                            text_pad_idx = model_engine.module.config.vocab_size - 1
                            speech_pad_idx = model_engine.module.config.speech_vocab_size - 1
                            
                            text_pad_emb = model_engine.module.embs[0].weight.data[text_pad_idx]
                            speech_pad_embs = [model_engine.module.embs[i].weight.data[speech_pad_idx] for i in range(1, model_engine.module.config.num_channels)]
                            
                            is_text_pad_zero = torch.all(text_pad_emb == 0)
                            are_speech_pads_zero = all(torch.all(emb == 0) for emb in speech_pad_embs)
                            
                            if is_text_pad_zero and are_speech_pads_zero:
                                logger.info(f"Step {global_step}: Padding embedding check PASSED. All padding vectors are zero.")
                            else:
                                logger.warning(f"Step {global_step}: Padding embedding check FAILED. Padding vectors have been updated.")
                        except Exception as e:
                            logger.error(f"Could not perform padding embedding check at step {global_step}: {e}")

                    # Save checkpoint
                    if global_step % args.save_steps == 0:
                        if args.ds_stage == 3 or args.ds_stage == 2:
                            save_checkpoint(model_engine, args.output_dir, epoch, step, logger)
                        
                except Exception as e:
                    logger.error(f"Error in training step {global_step}: {e}")
                    continue

            if is_main_process:
                pbar.close()
                
            # Save checkpoint at the end of each epoch
            if args.ds_stage == 3 or args.ds_stage == 2:
                epoch_checkpoint_dir = f"{args.output_dir}/epoch_{epoch}"
                os.makedirs(epoch_checkpoint_dir, exist_ok=True)
                print(f'saving checkpoint to {epoch_checkpoint_dir}')
                model_engine.save_checkpoint(epoch_checkpoint_dir)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise

    logger.info("--- Training Finished ---")
    model_engine.save_checkpoint(args.output_dir, tag="final_checkpoint")
    if is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main()