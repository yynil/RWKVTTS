import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
import deepspeed
import datasets
from datasets import load_dataset
import wandb
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, field
import logging
import json
from typing import Optional
import time  
import regex as re
from pathlib import Path
import numpy as np
from model.llm.rwkv_tts import RWKV7TTSModel
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Config, RWKV7Model, RWKV7ForCausalLM
from utils.tts_util import create_inputs_and_labels

logger = logging.getLogger(__name__)

@dataclass
class ScriptArguments:
    """Command line arguments for training script"""
    data_dir: str = field(
        default=None,
        metadata={"help": "Path to training data directory with JSONL files"}
    )
    text_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained text model"}
    )
    audio_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained audio model (required)"}
    )
    output_dir: str = field(
        default=None,
        metadata={"help": "Directory to save trained model"}
    )
    deepspeed_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to DeepSpeed config file"}
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
        metadata={"help": "Local rank for distributed training"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    wandb_project: str = field(
        default="rwkv-tts-training",
        metadata={"help": "Name of W&B project"}
    )
    wandb_run_name: str = field(
        default=None,
        metadata={"help": "Name of W&B run"}
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing"}
    )
    
    ds_stage: int = field(
        default=3,
        metadata={"help": "DeepSpeed stage"}
    )

    ds_param_offload : bool = field(
        default=True,
        metadata={"help": "DeepSpeed parameter offload"}
    )
    
    ds_optimizer_offload : bool = field(
        default=True,
        metadata={"help": "DeepSpeed optimizer offload"}
    )
    
    drop_out: float = field(
        default=0.02,
        metadata={"help": "drop out"}
    )
    
    ckpt_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model checkpoint file"}
    )

    max_tokens_k: int = field(
        default=32,
        metadata={"help": "Maximum tokens in K units (e.g. 32 means 32K tokens)"}
    )
    
    gradient_clipping: float = field(
        default=0.5,
        metadata={"help": "Gradient clipping threshold"}
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
    """配置优化器，只优化音频模型相关的参数"""
    lr_1x = set()
    lr_2x = set()
    lr_decay = set()    
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 只优化音频模型相关的参数，文本模型参数被冻结
        if 'text_lm_model' in n:
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
        optimizer = DeepSpeedCPUAdam(optim_groups, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-18, bias_correction=True, adamw_mode=adamw_mode, amsgrad=False,weight_decay=args.weight_decay)
    else:
        from deepspeed.ops.adam import FusedAdam
        optimizer = FusedAdam(optim_groups, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-18, bias_correction=True, adam_w_mode=adamw_mode, amsgrad=False, weight_decay=args.weight_decay)
  
    return optimizer

def save_checkpoint(model_engine, output_dir, epoch, step, logger):
    """Save model checkpoint"""
    if os.path.exists(output_dir):
        if model_engine.local_rank == 0:
            checkpoints = os.listdir(output_dir)
            #only list the directories   
            checkpoints = [f for f in checkpoints if os.path.isdir(os.path.join(output_dir, f))]
            #sort by creation time  
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

def update_learning_rate(optimizer, current_step, total_steps, warmup_steps, learning_rate, learning_rate_final, args, is_main_process):
    """更新优化器中每个参数组的学习率"""
    import math
    
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
            print(f'param_group: {param_group["name"]} lr: {param_group["lr"]} weight_decay: {param_group["weight_decay"]}, params: {param_group["params"]}')

def compute_gradient_norm(model_engine):
    """计算梯度范数"""
    total_norm = 0.0
    for p in model_engine.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

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



def train_step(model_engine, text_input_ids, text_attention_mask, audio_token_ids, audio_token_attention_mask, labels):
    """执行一步训练"""
    
    # 前向传播
    outputs = model_engine(
        text_input_ids=text_input_ids,
        text_attention_mask=text_attention_mask,
        audio_token_ids=audio_token_ids,
        audio_token_attention_mask=audio_token_attention_mask,
        labels=labels
    )
    
    loss = outputs.loss
    if torch.isnan(loss) or torch.isinf(loss):
        print(f'loss is nan or inf, loss: {loss}')
        print(f'outputs: {outputs}')
    
    return {
        'loss': loss
    }

def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Setup environment variables
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    is_main_process = local_rank == 0
    device = torch.device(f'cuda:{local_rank}')
    
    if is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    # Setup logging
    setup_logging(local_rank)
    logger = logging.getLogger(__name__)
    
    if is_main_process:
        logger.info(f"Arguments: {args}")
        logger.info(f"梯度裁剪阈值设置为: {args.gradient_clipping}")

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Initialize tokenizer
    if is_main_process:
        logger.info(f"Loading tokenizer from {args.text_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load dataset
    if args.data_dir is None:
        raise ValueError("data_dir must be set")
    
    logger.info(f"Loading dataset from JSONL files in: {args.data_dir}")
    jsonl_files = [str(p) for p in Path(args.data_dir).rglob("*.jsonl")]
    if not jsonl_files:
        raise ValueError(f"No .jsonl files found in {args.data_dir}")
    
    dataset = load_dataset("json", data_files=jsonl_files, split="train")
    
    # Setup data loading
    logger.info(f"Creating DataLoader with batch size {args.per_device_train_batch_size}, world size {world_size}")
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        seed=args.seed
    )
    
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
                "dump_state": False,
                # 添加梯度裁剪配置
                "gradient_clipping": args.gradient_clipping
            }
    
    # Load text model (只需要RWKV7Model，不需要lm_head)
    if is_main_process:
        logger.info(f"Loading text model from {args.text_model_path}")
    text_model = RWKV7Model.from_pretrained(args.text_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    
    # Load audio model from pretrained path
    if is_main_process:
        logger.info(f"Loading audio model from {args.audio_model_path}")
    audio_model = RWKV7ForCausalLM.from_pretrained(args.audio_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    
    # Create TTS model
    if is_main_process:
        logger.info("Creating TTS model")
    model = RWKV7TTSModel(text_model, audio_model)
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    
    if args.ckpt_file is not None:
        if is_main_process:
            logger.info(f"Loading checkpoint from {args.ckpt_file}")
        info = model.load_state_dict(torch.load(args.ckpt_file))
        if is_main_process:
            logger.info(f"Loaded checkpoint info: {info}")
    
    if is_main_process:
        logger.info(f'Enable gradient checkpointing: {args.gradient_checkpointing}')
    
    # 冻结文本模型参数
    for n, p in model.text_lm_model.named_parameters():
        p.requires_grad = False
        if is_main_process:
            print(f'冻结文本模型参数: {n}')
    
    if is_main_process:
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f'{n} requires grad: {p.requires_grad}')
        logger.info(f'start configuring optimizer')
    
    optimizer = configure_optimizer(model, args)
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config,
            model_parameters=model.parameters(),
            optimizer=optimizer
    )
   
    # 删除原始model
    del model
    

    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda x: x
    )
        
    # 删除输出目录
    if os.path.exists(args.output_dir) and model_engine.local_rank == 0:
        import shutil
        shutil.rmtree(args.output_dir)
    
    total_loss = 0.0
    total_steps = 0
    all_tokens = 0
    global_steps = 0

    print(f'warming up the model')
    model_engine.train()
    
    # 预热模型
    dummy_text_ids = torch.randint(0, 65536, (2, 20), device=device, dtype=torch.long)
    dummy_text_mask = torch.ones(2, 20, device=device, dtype=torch.long)
    dummy_audio_tokens = torch.randint(0, 12289, (2, 132), device=device, dtype=torch.long)  # 32 global + 100 semantic
    dummy_audio_mask = torch.ones(2, 132, device=device, dtype=torch.long)
    dummy_labels = torch.full((2, 132), -100, device=device, dtype=torch.long)
    # 设置语义token部分的标签（后100个位置）
    dummy_labels[:, -100:] = torch.randint(0, 8192, (2, 100), device=device, dtype=torch.long)
    
    outputs = model_engine(
        text_input_ids=dummy_text_ids,
        text_attention_mask=dummy_text_mask,
        audio_token_ids=dummy_audio_tokens,
        audio_token_attention_mask=dummy_audio_mask,
        labels=dummy_labels,
    )
    loss = outputs.loss
    print(f'warm up loss: {loss} device: {device}')
    loss = loss * 0.0
    model_engine.backward(loss)
    model_engine.step()
    print(f'model warmed up')
    
    del dummy_text_ids, dummy_text_mask, dummy_audio_tokens, dummy_audio_mask, dummy_labels
    torch.cuda.empty_cache()
    
    for epoch in range(args.num_epochs):
        model_engine.train()
        if is_main_process:
            update_time = time.time()
            logger.info(f"Epoch {epoch} starts training")
            from tqdm import tqdm
            pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")
        
        # 使用时间戳生成随机种子
        time_seed = int(time.time() * 1000) & 0xffffffff
        sampler.set_epoch(time_seed)
        eos_id = 8192
        for batch_idx, batch in enumerate(dataloader):
            # 更新学习率
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
            
            # 处理批次数据
            processed_batch = create_inputs_and_labels(batch, tokenizer,eos_id, device=device,is_properties_used=False,
                                                       is_global_tokens_predictable=False,
                                                        randomly_mark_phonems=True,random_mark_prob=0.5)
            
            if is_main_process and batch_idx == 0:
                print(f'text_input_ids shape: {processed_batch["text_input_ids"].shape}')
                print(f'text_attention_mask shape: {processed_batch["text_attention_mask"].shape}')
                print(f'audio_token_ids shape: {processed_batch["audio_token_ids"].shape}')
                print(f'audio_token_attention_mask shape: {processed_batch["audio_token_attention_mask"].shape}')
                print(f'labels length: {len(processed_batch["labels"])}')

            output = train_step(model_engine, **processed_batch)
            loss = output['loss']
            if is_main_process and batch_idx == 0:
                print(f'loss no properties: {loss}')
            
            # 反向传播
            model_engine.backward(loss)
            
            # 计算梯度范数并监控
            if global_steps % 100 == 0:
                grad_norm = compute_gradient_norm(model_engine)
                if is_main_process:
                    print(f"Step {global_steps}: Gradient norm = {grad_norm:.4f}")
                    # 如果梯度范数过大，给出警告
                    if grad_norm > 10.0:
                        print(f"Warning: Large gradient norm detected: {grad_norm:.4f}")
            
            model_engine.step()
            processed_batch = create_inputs_and_labels(batch, tokenizer,eos_id, device=device,is_properties_used=True,
                                                       is_global_tokens_predictable=True,
                                                        randomly_mark_phonems=True,random_mark_prob=0.5)
            if is_main_process and batch_idx == 0:
                print(f'text_input_ids shape: {processed_batch["text_input_ids"].shape}')
                print(f'text_attention_mask shape: {processed_batch["text_attention_mask"].shape}')
                print(f'audio_token_ids shape: {processed_batch["audio_token_ids"].shape}')
                print(f'audio_token_attention_mask shape: {processed_batch["audio_token_attention_mask"].shape}')
                print(f'labels length: {len(processed_batch["labels"])}')

            output = train_step(model_engine, **processed_batch)
            loss = output['loss']
            if is_main_process and batch_idx == 0:
                print(f'loss with properties: {loss}')
            global_steps += 1
            
            # 反向传播
            model_engine.backward(loss)
            
            # 计算梯度范数并监控
            if global_steps % 100 == 0:
                grad_norm = compute_gradient_norm(model_engine)
                if is_main_process:
                    print(f"Step {global_steps}: Gradient norm = {grad_norm:.4f}")
                    # 如果梯度范数过大，给出警告
                    if grad_norm > 10.0:
                        print(f"Warning: Large gradient norm detected: {grad_norm:.4f}")
            
            model_engine.step()

            if batch_idx % args.save_steps == 0 and batch_idx > 0:
                if args.ds_stage == 3 or args.ds_stage == 2:
                    save_checkpoint(model_engine, args.output_dir, epoch, batch_idx, logger)
            
            # 累计统计
            if is_main_process:
                elapsed_time = time.time()-update_time
                update_time = time.time()
                total_loss += loss.item()
                total_steps += 1
                
                # 计算平均值
                avg_loss = total_loss / total_steps
                tokens = processed_batch["text_input_ids"].shape[0] * processed_batch["text_input_ids"].shape[1]
                all_tokens += (2*tokens*world_size)
                kts =  (tokens*world_size) / elapsed_time / 1e3
                
                # 记录到wandb
                current_lr = optimizer.param_groups[0]['lr']
                log_metrics(optimizer, loss, avg_loss, epoch, total_steps, kts, all_tokens, current_lr)
                
                pbar.update(1)
                pbar.set_postfix({
                    'loss': loss.item(),
                    'avg_loss': avg_loss,
                    'lr': current_lr,
                    "kts": kts,
                    "steps": total_steps,
                    "epoch": epoch
                })
        
        # 保存检查点
        if args.ds_stage == 3 or args.ds_stage == 2:
            epoch_checkpoint_dir = f"{args.output_dir}/epoch_{epoch}"
            os.makedirs(epoch_checkpoint_dir, exist_ok=True)
            print(f'saving checkpoint to {epoch_checkpoint_dir}')
            model_engine.save_checkpoint(epoch_checkpoint_dir)
        
        if is_main_process:
            pbar.close()
    
    # 训练结束后关闭wandb
    if is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main()
