from ast import mod
from calendar import c
import os
from turtle import up
import torch
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
from torch.utils.data import DataLoader, DistributedSampler
import deepspeed
import datasets
import wandb
from transformers import HfArgumentParser, AutoTokenizer,AutoModelForCausalLM
from dataclasses import dataclass, field
import logging
import json
from typing import Optional
from functools import partial
import time  
import regex as re
from data.utils.llm_dataset import load_jsonl_dataset, collate_fn
from model.llm.llm import RWKV7LM
from train_scripts.train_functions import train_step
logger = logging.getLogger(__name__)
@dataclass
class ScriptArguments:
    """Command line arguments for training script"""
    data_file: str = field(
        default=None,
        metadata={"help": "Path to training data file (JSONL format)"}
    )
    model_name: str = field(
        default=None,
        metadata={"help": "Path or name of pretrained model"}
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
        default="grpo-training",
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
    
    chunk_size : int = field(
        default=1024,
        metadata={"help": "chunk size"}
    )
    
    batch_chunk_size: int = field(
        default=2,
        metadata={"help": "batch chunk size"}
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
    speech_token_size: int  = field(
        default=6561,
        metadata={"help": "speech token size"}
    )
    
    drop_out: float = field(
        default=0.02,
        metadata={"help": "drop out"}
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
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lr_1x.add(n)

    lr_1x = sorted(list(lr_1x))
    param_dict = {n: p for n, p in model.named_parameters()}
    
    optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

    if args.ds_optimizer_offload:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        optimizer = DeepSpeedCPUAdam(optim_groups, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-18, bias_correction=True, adamw_mode=True, amsgrad=False,weight_decay=args.weight_decay)
    else:
        from deepspeed.ops.adam import FusedAdam
        optimizer = FusedAdam(optim_groups, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-18, bias_correction=True, adam_w_mode=True, amsgrad=False, weight_decay=args.weight_decay)
  
    return optimizer

def save_checkpoint(model_engine, output_dir, epoch, step,logger):
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
def get_lr_scheduler(optimizer, total_steps, warmup_steps, learning_rate, learning_rate_final):
    """Create a linear learning rate scheduler that goes from learning_rate to learning_rate_final"""
    from transformers import get_linear_schedule_with_warmup
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # 在预热阶段，从0线性增加到learning_rate
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # 预热后，从learning_rate线性减少到learning_rate_final
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(learning_rate_final / learning_rate, 1.0 - progress * (1.0 - learning_rate_final / learning_rate))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Setup environment variables
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    is_main_process = local_rank == 0
    device = torch.device(f'cuda:{local_rank}')
    
    # Setup logging
    setup_logging(local_rank)
    logger = logging.getLogger(__name__)
    
    if is_main_process:
        logger.info(f"Arguments: {args}")

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Initialize tokenizer
    if is_main_process:
        logger.info(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name,trust_remote_code=True)
    tokenizer.add_special_tokens({'pad_token': '<|rwkv_tokenizer_end_of_text|>'})
    
    # Load dataset
    if is_main_process:
        logger.info(f"Loading dataset from {args.data_file}")
    dataset = load_jsonl_dataset(args.data_file,tokenizer)
    
    # Setup data loading
    if is_main_process:
        logger.info(f"Creating DataLoader with batch size {args.per_device_train_batch_size}, world size {world_size}")
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        seed=args.seed
    )
    
    data_collator = partial(collate_fn,tokenizer=tokenizer,max_length=args.max_length,pad_to_max_length=False,drop_prompt_audio_rate=0.5)
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=data_collator
    )
    
    # Load DeepSpeed config
    if args.deepspeed_config:
        if is_main_process:
            logger.info(f"Loading DeepSpeed config from {args.deepspeed_config}")
        with open(args.deepspeed_config, 'r') as f:
            ds_config = json.load(f)
    else:
        # Default DeepSpeed config is using ZeRO-3 with CPU offload
        if is_main_process:
            logger.info("Using default DeepSpeed config")
        train_batch_size = args.per_device_train_batch_size * world_size* 1
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
                    "overlap_comm": True,
                    "contiguous_gradients": True
                },
                "zero_force_ds_cpu_initialization": True,
                "gradient_checkpointing": args.gradient_checkpointing,
                "dump_state": True
            }
        
    #Init model with deepspeed
    if is_main_process:
        logger.info(f"Initializing model with DeepSpeed config")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16,trust_remote_code=True)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    llm_input_size = model.config.hidden_size
    llm_output_size = model.config.hidden_size
    model  = RWKV7LM(llm_input_size,llm_output_size,args.speech_token_size,model,None)
    model.train()
    if is_main_process:
        logger.info(f'Enable gradient checkpointing: {args.gradient_checkpointing}')
    for n,p in model.named_parameters():
        p.requires_grad = True
    if is_main_process:
        for n,p in model.named_parameters():
            print(f'{n} requires grad: {p.requires_grad}')
        logger.info(f'start configuring optimizer')
    optimizer = configure_optimizer(model, args)
    # Initialize DeepSpeed for main model
    model_ds_config = ds_config.copy()
    if not args.ds_param_offload:
        del model_ds_config["zero_optimization"]["offload_param"]
    if not args.ds_optimizer_offload:
        del model_ds_config["zero_optimization"]["offload_optimizer"]
        
    # 在初始化DeepSpeed之前计算总步数
    total_steps = len(dataloader) * args.num_epochs
    
    # 创建自定义的学习率调度器
    lr_scheduler = get_lr_scheduler(
        optimizer, 
        total_steps,
        args.warmup_steps,
        args.learning_rate,
        args.learning_rate_final
    )
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            config=model_ds_config,
            model_parameters=model.parameters(),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler
    )
    if is_main_process:
        logger.info("Model initialized")
    del model
    if is_main_process:
        from tqdm import tqdm
        pbar = tqdm(total=len(dataloader))
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    #delete the output_dir if it exists
    if os.path.exists(args.output_dir) and model_engine.local_rank == 0:
        import shutil
        shutil.rmtree(args.output_dir)
    total_loss = 0.0
    total_steps = 0
    total_acc = 0.0
    all_tokens = 0
    for epoch in range(args.num_epochs):
        if is_main_process:
            update_time = time.time()
            logger.info(f"Epoch {epoch} starts training")
        # 使用时间戳生成随机种子
        time_seed = int(time.time() * 1000) & 0xffffffff  # 获取毫秒级时间戳并转换为32位整数
        sampler.set_epoch(time_seed)  # 使用时间戳作为种子
        
        for batch_idx,batch in enumerate(dataloader):
            if is_main_process:
                speech_token_shape = batch['speech_token'].shape
                text_token_shape = batch['text_token'].shape
                logger.debug(f'speech_token_shape: {speech_token_shape} text_token_shape: {text_token_shape} at batch_idx: {batch_idx}')
            skip = batch['skip']
            if skip:
                all_length = batch['text_token'].shape[1] + batch['speech_token'].shape[1]
                if all_length > args.max_length:
                    #truncate the sppech_token first
                    truncated_length = args.max_length - batch['speech_token'].shape[1]
                    speech_token = batch['speech_token']
                    batch['speech_token'] = speech_token[:,:truncated_length]
            batch.pop('skip')
            output = train_step(model_engine,batch)
            loss = output['loss']
            acc = output['acc']
            if is_main_process:
                logger.debug(f'loss: {loss} acc: {acc}')
                
            # 首先检测 NaN
            is_nan_loss = torch.isnan(loss) or torch.isinf(loss)
            # 确保所有进程获得相同的 is_nan_loss 值
            is_nan_loss_tensor = torch.tensor([1.0 if is_nan_loss else 0.0], device=model_engine.device)
            # 所有进程同步这个张量以获取一致决策
            torch.distributed.all_reduce(is_nan_loss_tensor, op=torch.distributed.ReduceOp.MAX)
            is_nan_loss = bool(is_nan_loss_tensor.item())

            if is_nan_loss:
                # 使用一个安全的替代 loss 进行 backward
                # 这个 loss 不会影响模型（乘以0），但会确保所有节点都执行 backward
                logger.info(f"NaN loss detected at batch {batch_idx}, using safe zero loss instead")
                logger.info(f'batch data is {batch}')
                safe_loss = loss * 0.0 
                if is_main_process:
                    logger.warning(f"NaN loss detected at batch {batch_idx}, using safe zero loss instead")
                    wandb.log({
                        "nan_detected": 1,
                        "epoch": epoch,
                        "step": total_steps
                    })
                
                # 所有节点都执行 backward，但使用的是零梯度
                model_engine.backward(safe_loss)
                model_engine.step()  # 这步实际上不会改变参数，因为梯度是零
            else:
                # 正常情况，使用实际 loss
                model_engine.backward(loss)
                model_engine.step()
                
            if batch_idx % args.save_steps == 0 and batch_idx > 0:
                if args.ds_stage == 3 or args.ds_stage == 2:
                    save_checkpoint(model_engine, args.output_dir, epoch, batch_idx,logger)
            # 累计统计
            if is_main_process:
                elapsed_time = time.time()-update_time
                total_loss += loss.item()
                total_acc += acc.item()
                total_steps += 1
                
                # 计算平均值
                avg_loss = total_loss / total_steps
                avg_acc = total_acc / total_steps
                tokens = (batch['speech_token'].shape[1]+batch['text_token'].shape[1])*args.per_device_train_batch_size*world_size
                all_tokens += tokens
                kts = tokens / elapsed_time / 1e3
                # 记录到wandb
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({
                    "loss": loss.item(),
                    "avg_loss": avg_loss,
                    "epoch": epoch,
                    "step": total_steps,
                    "acc": acc.item(),
                    "avg_acc": avg_acc,
                    "KT/s": kts,
                    "Gtokens": all_tokens/1e9,
                    "learning_rate": current_lr
                })
                
                pbar.update(1)
                pbar.set_postfix({
                    'loss': loss.item(),
                    'avg_loss': avg_loss,
                    'acc': acc.item(),
                    'avg_acc': avg_acc,
                    'lr': current_lr
                })
        #save checkpoint at the end of each epoch
        # if (args.ds_stage != 3 and is_main_process) or (args.ds_stage == 3):
        if  args.ds_stage == 3 or args.ds_stage == 2:
            epoch_checkpoint_dir = f"{args.output_dir}/epoch_{epoch}"
            if not os.path.exists(epoch_checkpoint_dir):
                os.makedirs(epoch_checkpoint_dir)
            print(f'saving checkpoint to {epoch_checkpoint_dir}')
            model_engine.save_checkpoint(epoch_checkpoint_dir)
    # 训练结束后关闭wandb
    if is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main()