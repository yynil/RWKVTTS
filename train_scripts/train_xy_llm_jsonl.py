import multiprocessing as mp
mp.set_start_method('spawn', force=True)
from datasets import load_dataset
import os
import torch
from torch.utils.data import DataLoader, DistributedSampler, Dataset, ConcatDataset
import deepspeed
from transformers import HfArgumentParser, AutoTokenizer
from dataclasses import dataclass, field
import logging
import json
from typing import Optional, List, Dict
import time
import numpy as np
import random
import math
import wandb
import glob

# Custom model and data components
from model.llm.xy_llm import RWKV7XYLM
from utils.xy_data_processor import XYDataProcessor

logger = logging.getLogger(__name__)

@dataclass
class ScriptArguments:
    """Command line arguments for training the RWKV7XYLM model from JSONL data."""
    model_name_or_path: str = field(
        metadata={"help": "Path to the converted RWKV7XYLM model directory."}
    )
    jsonl_path: str = field(
        metadata={"help": "Path to the directory containing training .jsonl files."}
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
        default="rwkv7-xy-lm-training-jsonl", metadata={"help": "Name of W&B project."}
    )
    wandb_run_name: Optional[str] = field(
        default=None, metadata={"help": "Name of W&B run."}
    )
    # --- System and Environment ---
    local_rank: int = field(default=-1, metadata={"help": "Local rank for distributed training."})
    seed: int = field(default=42, metadata={"help": "Random seed."})

class JsonlDataset(Dataset):
    """
    A PyTorch Dataset for reading large JSONL files without loading them fully into memory.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.line_offsets = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            offset = 0
            for line in f:
                self.line_offsets.append(offset)
                offset += len(line.encode('utf-8'))
        self.file = None

    def __len__(self):
        return len(self.line_offsets)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = open(self.file_path, 'r', encoding='utf-8')
        self.file.seek(self.line_offsets[idx])
        line = self.file.readline()
        return json.loads(line)

    def __del__(self):
        if self.file:
            self.file.close()

def collate_fn(batch: List[Dict]) -> Dict[str, List]:
    """A simple collate function that groups data by keys."""
    keys = batch[0].keys()
    return {key: [d[key] for d in batch] for key in keys}

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def configure_optimizer(model, args):
    """Configure optimizer with parameter grouping"""
    lr_1x, lr_2x, lr_decay = set(), set(), set()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'attn.w_lora.lora.2.bias' in n:
            lr_2x.add(n)
        elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0) and (".weight" in n) and ("lora" not in n):
            lr_decay.add(n)
        else:
            lr_1x.add(n)

    param_dict = {n: p for n, p in model.named_parameters()}
    optim_groups = [
        {"params": [param_dict[n] for n in sorted(list(lr_1x))], "weight_decay": 0.0, "my_lr_scale": 1.0, "name": "lr_1x"},
        {"params": [param_dict[n] for n in sorted(list(lr_2x))], "weight_decay": 0.0, "my_lr_scale": 2.0, "name": "lr_2x"}
    ]
    if args.weight_decay > 0:
        optim_groups.append({"params": [param_dict[n] for n in sorted(list(lr_decay))], "weight_decay": args.weight_decay, "my_lr_scale": 1.0, "name": "lr_decay"})
        adamw_mode = True
    else:
        adamw_mode = False
    
    if args.ds_optimizer_offload:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        return DeepSpeedCPUAdam(optim_groups, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-18, bias_correction=True, adamw_mode=adamw_mode, amsgrad=False, weight_decay=args.weight_decay)
    else:
        from deepspeed.ops.adam import FusedAdam
        return FusedAdam(optim_groups, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-18, bias_correction=True, adam_w_mode=adamw_mode, amsgrad=False, weight_decay=args.weight_decay)

def update_learning_rate(optimizer, current_step, total_steps, warmup_steps, lr_start, lr_final, args, is_main_process):
    """Update learning rate for each parameter group."""
    if current_step < warmup_steps:
        base_lr = lr_start * (0.01 + 0.99 * current_step / warmup_steps)
    else:
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = max(0, min(1, progress))
        lr_final_factor = lr_final / lr_start
        base_lr = lr_start * ((0.5 + lr_final_factor / 2) + (0.5 - lr_final_factor / 2) * math.cos(math.pi * progress))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr * param_group.get('my_lr_scale', 1.0)

def log_metrics(optimizer, loss, avg_loss, epoch, step, kts, all_tokens, current_lr):
    """Log metrics to W&B."""
    wandb.log({
        "loss": loss.item(), "avg_loss": avg_loss, "epoch": epoch, "step": step,
        "KT/s": kts, "Gtokens": all_tokens / 1e9, "learning_rate": current_lr
    })

def save_checkpoint(model_engine, output_dir, epoch, step):
    """Save a DeepSpeed checkpoint."""
    checkpoint_dir = f"{output_dir}/epoch_{epoch}_step_{step}"
    model_engine.save_checkpoint(checkpoint_dir)
    logger.info(f"Saved checkpoint to {checkpoint_dir}")

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    is_main_process = local_rank == 0
    
    set_seed(args.seed)
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()

    if is_main_process:
        logging.basicConfig(level=logging.INFO)
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    logger.info(f"Starting training with arguments: {args}")

    text_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    model = RWKV7XYLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.zero_embs()
    logger.info("Padding embeddings have been zeroed out.")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    data_processor = XYDataProcessor(
        text_tokenizer=text_tokenizer,
        num_channels=model.config.num_channels,
        text_shift_size=text_tokenizer.encode("[SP0]")[0],
        speech_vocab_size=model.config.speech_vocab_size
    )

    logger.info(f"Loading dataset from directory: {args.jsonl_path}")
    jsonl_files = glob.glob(os.path.join(args.jsonl_path, '**/*.jsonl'), recursive=True)
    if not jsonl_files:
        raise ValueError(f"No .jsonl files found in {args.jsonl_path}")
    
    logger.info(f"Found {len(jsonl_files)} JSONL files.")
    
    dataset = load_dataset("json", data_files=jsonl_files, split="train")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    
    dataloader = DataLoader(
        dataset, batch_size=args.per_device_train_batch_size, sampler=sampler,
        num_workers=4, pin_memory=True, collate_fn=collate_fn, persistent_workers=True
    )

    optimizer = configure_optimizer(model, args)
    
    ds_config = {
        "train_batch_size": args.per_device_train_batch_size * world_size,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": args.ds_stage,
            "offload_param": {"device": "cpu", "pin_memory": True} if args.ds_param_offload else None,
            "offload_optimizer": {"device": "cpu", "pin_memory": True} if args.ds_optimizer_offload else None,
        },
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    if args.deepspeed_config:
        with open(args.deepspeed_config, 'r') as f:
            ds_config = json.load(f)

    model_engine, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_config)

    total_steps = len(dataloader) * args.num_epochs
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    logger.info("*** Starting Training ***")
    global_step, total_loss, all_tokens = 0, 0.0, 0
    
    try:
        for epoch in range(args.num_epochs):
            model_engine.train()
            sampler.set_epoch(epoch)
            
            if is_main_process:
                update_time = time.time()
                tokens_since_last_log = 0
                from tqdm import tqdm
                pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")
            
            for step, raw_batch in enumerate(dataloader):
                batch = data_processor.process_batch(raw_batch)
                if not batch: 
                    logger.warning("Empty batch received, skipping...")
                    continue

                if args.max_tokens_per_round:
                    B, T, C = batch['input_ids'].shape
                    max_tokens = args.max_tokens_per_round * 1024
                    if B * T > max_tokens:
                        max_batch_size = max_tokens // T
                        batch['input_ids'] = batch['input_ids'][:max_batch_size]
                        batch['labels'] = batch['labels'][:max_batch_size]
                        batch['attention_mask'] = batch['attention_mask'][:max_batch_size]
                        logger.warning(f"Batch token size ({B*T}) > max. Cut to {max_batch_size*T} tokens. from {B} to {max_batch_size}")
                        

                update_learning_rate(optimizer, global_step, total_steps, args.warmup_steps, args.learning_rate, args.learning_rate_final, args, is_main_process)

                try:
                    input_ids = batch['input_ids'].to(model_engine.device)
                    labels = batch['labels'].to(model_engine.device)
                    attention_mask = batch.get('attention_mask').to(model_engine.device)

                    outputs = model_engine(input_ids=input_ids, labels=labels, attention_mask=attention_mask, return_dict=True, use_cache=False)
                    loss = outputs.loss

                    model_engine.backward(loss)
                    model_engine.step()
                    global_step += 1

                    if is_main_process:
                        total_loss += loss.item()
                        batch_tokens = input_ids.numel()
                        tokens_since_last_log += batch_tokens * world_size
                        all_tokens += batch_tokens * world_size

                        if global_step % args.logging_steps == 0:
                            elapsed_time = time.time() - update_time
                            kts = (tokens_since_last_log / elapsed_time / 1e3) if elapsed_time > 0 else 0.0
                            avg_loss = total_loss / args.logging_steps
                            current_lr = optimizer.param_groups[0]['lr']
                            
                            logger.info(f"Epoch: {epoch}, Step: {global_step}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}, KT/s: {kts:.2f}")
                            log_metrics(optimizer, loss, avg_loss, epoch, global_step, kts, all_tokens, current_lr)
                            
                            total_loss, tokens_since_last_log = 0.0, 0
                            update_time = time.time()
                        
                        pbar.update(1)
                        pbar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})

                    if global_step % args.save_steps == 0:
                        save_checkpoint(model_engine, args.output_dir, epoch, global_step)
                        
                except Exception as e:
                    logger.error(f"Error in training step {global_step}: {e}")
                    continue

            if is_main_process:
                pbar.close()
            save_checkpoint(model_engine, args.output_dir, epoch, "epoch_end")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        raise

    logger.info("--- Training Finished ---")
    model_engine.save_checkpoint(args.output_dir, tag="final_checkpoint")
    if is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main()
