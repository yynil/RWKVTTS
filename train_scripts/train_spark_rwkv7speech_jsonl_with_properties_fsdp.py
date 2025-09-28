"""
This script is a modification of train_spark_rwkv7speech_jsonl_with_properties.py,
adapted to use PyTorch's Fully Sharded Data Parallelism (FSDP) instead of DeepSpeed.

Key changes:
- Replaced DeepSpeed initialization with FSDP wrapper.
- Swapped DeepSpeed optimizer with standard PyTorch AdamW.
- Implemented FSDP-compatible checkpoint saving.
- Adjusted the training loop to use standard PyTorch backward pass and optimizer steps.
- Removed DeepSpeed-specific configurations and arguments.
"""
from ast import mod
from calendar import c
import os
from turtle import up
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig, ShardingStrategy, CPUOffload, MixedPrecision   
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy,size_based_auto_wrap_policy
from transformers.models.rwkv.modeling_rwkv import RwkvBlock
import functools    

import datasets
from datasets import load_dataset
import wandb
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, field
import logging
import json
from typing import Optional
from functools import partial
import time  
import regex as re
from sparktts.models.audio_tokenizer import BiCodecTokenizer
import soundfile as sf
from pathlib import Path
import numpy as np
from inference.rwkv7speech_inference import create_inputs

logger = logging.getLogger(__name__)
@dataclass
class ScriptArguments:
    """Command line arguments for training script"""
    data_dir: str = field(
        default=None,
        metadata={"help": "Path to training data directory with JSONL files"}
    )
    model_name: str = field(
        default=None,
        metadata={"help": "Path or name of pretrained model"}
    )
    spark_model_dir: str = field(
        default=None,
        metadata={"help": "Path to Spark model directory for audio tokenizer"}
    )
    output_dir: str = field(
        default=None,
        metadata={"help": "Directory to save trained model"}
    )
    num_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Training batch size per device"}
    )
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Learning rate"}
    )
    learning_rate_final: float = field(
        default=1e-6,
        metadata={"help": "Final learning rate at the end of training"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay"}
    )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "Number of warmup steps"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum length of input sequence"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Number of steps between logging"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Number of steps between saving checkpoints"}
    )
    local_rank: int = field(
        default=-1,
        metadata={"help": "Local rank for distributed training (handled by FSDP)"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    wandb_project: str = field(
        default="spark-training-fsdp",
        metadata={"help": "Name of W&B project"}
    )
    wandb_run_name: str = field(
        default=None,
        metadata={"help": "Name of W&B run"}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing"}
    )
    drop_out: float = field(
        default=0.02,
        metadata={"help": "drop out"}
    )
    ckpt_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model checkpoint file"}
    )
    demo_dir: str = field(
        default="/home/yueyulin/github/RWKVTTS/demos/",
        metadata={"help": "Directory containing demo audio files for generation"}
    )
    demo_every_steps: int = field(
        default=1000,
        metadata={"help": "Generate demo every N steps"}
    )
    max_tokens_k: int = field(
        default=32,
        metadata={"help": "Maximum tokens in K units (e.g. 32 means 32K tokens)"}
    )

def setup_logging(local_rank):
    """Configure logging"""
    if local_rank <= 0:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if 'LOG_LEVEL' not in os.environ else os.environ['LOG_LEVEL'],
        )

def configure_optimizer(model, args):
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
    
    from torch.optim import AdamW
    optimizer = AdamW(optim_groups, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8)
    return optimizer

def save_checkpoint(model, output_dir, epoch, step, logger, rank):
    """Save model checkpoint with FSDP"""
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        state_dict = model.state_dict()

    if rank == 0:
        if os.path.exists(output_dir):
            checkpoints = os.listdir(output_dir)
            checkpoints = [f for f in checkpoints if os.path.isdir(os.path.join(output_dir, f))]
            checkpoints.sort(key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
            if len(checkpoints) > 2:
                print(f'deleting older checkpoints {checkpoints[0]}')
                import shutil
                shutil.rmtree(os.path.join(output_dir, checkpoints[0]))    
        
        output_dir = f"{output_dir}/epoch_{epoch}_step_{step}"
        print(f'saving checkpoint to {output_dir}')
        os.makedirs(output_dir, exist_ok=True)
        
        model_save_path = os.path.join(output_dir, 'pytorch_model.bin')
        torch.save(state_dict, model_save_path)

def update_learning_rate(optimizer, current_step, total_steps, warmup_steps, learning_rate, learning_rate_final,args,is_main_process):
    """更新优化器中每个参数组的学习率"""
    import math
    
    if current_step < warmup_steps:
        base_lr = learning_rate * (0.01 + 0.99 * current_step / warmup_steps)
    else:
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = max(0, min(1, progress))
        lr_final_factor = learning_rate_final / learning_rate
        base_lr = learning_rate * ((0.5 + lr_final_factor / 2) + (0.5 - lr_final_factor / 2) * math.cos(math.pi * progress))
    
    for param_group in optimizer.param_groups:
        if param_group.get('weight_decay', 0) > 0:
            param_group['weight_decay'] = args.weight_decay
        lr_scale = param_group.get('my_lr_scale', 1.0)
        param_group['lr'] = base_lr * lr_scale
        if is_main_process and current_step % 100 == 0:
            print(f'param_group: {param_group["name"]} lr: {param_group["lr"]} weight_decay: {param_group["weight_decay"]}')

def log_metrics(optimizer, loss, avg_loss, epoch, total_steps, kts, all_tokens, current_lr):
    """记录训练指标到 wandb"""
    wandb.log({
        "loss": loss.item(),
        "avg_loss": avg_loss,
        "epoch": epoch,
        "step": total_steps,
        "KT/s": kts,
        "Gtokens": all_tokens/1e9,
        "learning_rate": current_lr
    })
    
    for param_group in optimizer.param_groups:
        params = param_group['params']
        total_params = sum(p.numel() for p in params)
        wandb.log({
            f"lr_group_{param_group.get('name', 'default')}": param_group['lr'],
            f"wd_group_{param_group.get('name', 'default')}": param_group.get('weight_decay', 0),
            f"params_count_{param_group.get('name', 'default')}": total_params
        })

def train_step(model, input_embs,labels,cu_seqlens=None,attention_mask=None):
    """执行一步训练"""
    if cu_seqlens is not None:
        outputs = model(inputs_embeds=input_embs, cu_seqlens=cu_seqlens, labels=labels,use_cache=False)
    else:
        outputs = model(inputs_embeds=input_embs, attention_mask=attention_mask, labels=labels,use_cache=False)
    loss = outputs.loss
    if torch.isnan(loss) or torch.isinf(loss):
        print(f'loss is nan or inf, loss: {loss}')
        print(f'outputs: {outputs}')
    return {'loss': loss}

from utils.multiple_jsonl import create_inputs_and_labels_with_properties,create_inputs_and_labels_with_properties_culens

def main():
    # 设置环境变量以避免某些兼容性问题
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # FSDP setup
    local_rank = int(os.environ.get("LOCAL_RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    rank = int(os.environ.get("RANK"))
    print(f'local_rank: {local_rank}, world_size: {world_size}, rank: {rank}')
    dist.init_process_group("nccl")
    is_main_process = local_rank == 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    setup_logging(local_rank)
    logger = logging.getLogger(__name__)
    
    if is_main_process:
        logger.info(f"Arguments: {args}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    if is_main_process:
        logger.info(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if args.data_dir is None:
        raise ValueError("data_dir must be set")
    
    logger.info(f"Loading dataset from JSONL files in: {args.data_dir}")
    jsonl_files = [str(p) for p in Path(args.data_dir).rglob("*.jsonl")]
    if not jsonl_files:
        raise ValueError(f"No .jsonl files found in {args.data_dir}")
    
    dataset = load_dataset("json", data_files=jsonl_files, split="train")
    
    logger.info(f"Creating DataLoader with batch size {args.per_device_train_batch_size}, world size {world_size}")
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed
    )
    
    if is_main_process:
        logger.info(f"Initializing model")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    eos_token_id = model.config.vocab_size - 1
    hidden_size = model.config.hidden_size
    
    if args.ckpt_file is not None:
        if is_main_process:
            logger.info(f"Loading checkpoint from {args.ckpt_file}")
        info = model.load_state_dict(torch.load(args.ckpt_file, map_location='cpu'))
        if is_main_process:
            logger.info(f"Loaded checkpoint info: {info}")

    if is_main_process:
        logger.info(f'Enable gradient checkpointing: {args.gradient_checkpointing}')
    

    torch.cuda.set_device(local_rank)
    # FSDP wrapping
    def custom_wrap_policy(module, recurse, nonwrapped_numel):
        # 排除 RwkvBlock 和某些关键模块
        if isinstance(module, RwkvBlock):
            return False
        return True
    
    

    model = FSDP(
        model,
        auto_wrap_policy=custom_wrap_policy,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        cpu_offload=CPUOffload(offload_params=True),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
    )
    if rank == 0:
        print(f'model: {model}')

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if is_main_process:
        logger.info(f'start configuring optimizer')
    optimizer = configure_optimizer(model, args)
    
    def collate_fn(batch):
        keys = batch[0].keys()
        return {key: [d[key] for d in batch] for key in keys}

    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
        
    if os.path.exists(args.output_dir) and is_main_process:
        import shutil
        shutil.rmtree(args.output_dir)
    
    total_loss = 0.0
    total_steps = 0
    all_tokens = 0
    global_steps = 0

    print(f'warming up the model')
    model.train()
    optimizer.zero_grad()
    
    # 使用更简单的 warmup 测试
    try:
        dummy_input_embs = torch.randn(2, 512, hidden_size, device=device, dtype=torch.bfloat16,requires_grad=True)
        dummy_attn_mask = torch.ones(2, 512, device=device, dtype=torch.long)
        dummy_labels = torch.randint(0, 1000, (2, 512), device=device, dtype=torch.long)
        
        outputs = model(inputs_embeds=dummy_input_embs, attention_mask=dummy_attn_mask, labels=dummy_labels, use_cache=False)
        loss = outputs.loss
        print(f'warm up loss: {loss} device: {device}')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f'model warmed up successfully')
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'warmup failed: {e}')
        print('continuing without warmup...')
        exit(1)
    finally:
        try:
            del dummy_input_embs, dummy_attn_mask, dummy_labels, loss, outputs
        except:
            pass
        torch.cuda.empty_cache()

    for epoch in range(args.num_epochs):
        model.train()
        if is_main_process:
            update_time = time.time()
            logger.info(f"Epoch {epoch} starts training")
            from tqdm import tqdm
            pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")
        
        time_seed = int(time.time() * 1000) & 0xffffffff
        sampler.set_epoch(time_seed)
        
        for batch_idx, batch in enumerate(dataloader):
            update_learning_rate(
                optimizer,
                global_steps,
                len(dataloader) * args.num_epochs,
                args.warmup_steps,
                args.learning_rate,
                args.learning_rate_final,
                args,
                is_main_process
            )
            
            processed_batch = create_inputs_and_labels_with_properties_culens(batch, tokenizer, model, eos_token_id, device)
            
            maxium_tokens = args.max_tokens_k * 1024
            bsz = processed_batch["cu_seqlens"].shape[0]
            adjusted_length = processed_batch["cu_seqlens"][bsz-1]
            while bsz > 0 and adjusted_length > maxium_tokens:
                bsz -= 1
                adjusted_length = processed_batch["cu_seqlens"][bsz-1]
            if bsz < processed_batch["cu_seqlens"].shape[0]:
                print(f'shrink the batch size from {processed_batch["cu_seqlens"].shape[0]} to {bsz}')
                processed_batch["input_embs"] = processed_batch["input_embs"][:,:adjusted_length,:]
                processed_batch["labels"] = processed_batch["labels"][:,:adjusted_length]
                processed_batch["cu_seqlens"] = processed_batch["cu_seqlens"][:bsz]
            
            # current_batch_size, current_batch_seq_len, _ = processed_batch["input_embs"].shape
            # max_batch_size = maxium_tokens // current_batch_seq_len
            # if max_batch_size < current_batch_size:
            #     print(f'max_batch_size < current_batch_size, max_batch_size: {max_batch_size}, current_batch_size: {current_batch_size} shrink the batch size')
            #     processed_batch["input_embs"] = processed_batch["input_embs"][:max_batch_size]
            #     processed_batch["labels"] = processed_batch["labels"][:max_batch_size]
            #     processed_batch["attention_mask"] = processed_batch["attention_mask"][:max_batch_size]

            # if is_main_process and batch_idx == 0:
            #     print(f'input_embs shape: {processed_batch["input_embs"].shape}')
            #     print(f'labels shape: {processed_batch["labels"].shape}')
            #     print(f'attention_mask shape: {processed_batch["attention_mask"].shape}')

            optimizer.zero_grad()
            output = train_step(model, **processed_batch)
            loss = output['loss']
            loss.backward()
            optimizer.step()
            
            if is_main_process and batch_idx == 0:
                print(f'loss: {loss}')
            global_steps += 1
            
            if batch_idx % args.save_steps == 0 and batch_idx > 0:
                save_checkpoint(model, args.output_dir, epoch, batch_idx, logger, local_rank)
            
            if is_main_process:
                elapsed_time = time.time() - update_time
                update_time = time.time()
                total_loss += loss.item()
                total_steps += 1
                
                avg_loss = total_loss / total_steps
                tokens = processed_batch["input_embs"].shape[0] * processed_batch["input_embs"].shape[1]
                all_tokens += tokens
                kts = tokens / elapsed_time / 1e3
                
                current_lr = optimizer.param_groups[0]['lr']
                log_metrics(optimizer, loss, avg_loss, epoch, total_steps, kts, all_tokens, current_lr)
                
                pbar.update(1)
                pbar.set_postfix({
                    'loss': loss.item(),
                    'avg_loss': avg_loss,
                    'lr': current_lr
                })
        
        epoch_checkpoint_dir = f"{args.output_dir}/epoch_{epoch}"
        save_checkpoint(model, epoch_checkpoint_dir, epoch, 'final', logger, local_rank)
        
        if is_main_process:
            pbar.close()
    
    if is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main()
