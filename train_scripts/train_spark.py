from ast import mod
from calendar import c
import os
from turtle import up
import torch
from torch.utils.data import DataLoader, DistributedSampler
import deepspeed
import datasets
import wandb
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, field
import logging
import json
from typing import Optional
from functools import partial
import time  
import regex as re
from data.utils.spark_dataset import load_spark_jsonl_dataset, collate_fn,convert_to_tts_format
from sparktts.models.audio_tokenizer import BiCodecTokenizer
import soundfile as sf
from pathlib import Path
logger = logging.getLogger(__name__)
demo_text = "我们都是来自同一个地球，同一个梦想！同一片蓝天下！"
def generate_tts_text(text,global_str):
    formatted_text = f"<tts><text_start>{text}<text_end><global_start>{global_str}<global_end><sementic_start>"
    return formatted_text
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
    spark_model_dir: str = field(
        default=None,
        metadata={"help": "Path to Spark model directory for audio tokenizer"}
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
        default="spark-training",
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

    demo_dir: str = field(
        default="/home/yueyulin/github/RWKVTTS/demos/",
        metadata={"help": "Directory containing demo audio files for generation"}
    )

    demo_every_steps: int = field(
        default=1000,
        metadata={"help": "Generate demo every N steps"}
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

def train_step(model_engine, batch):
    """执行一步训练"""
    # 将数据移动到正确的设备
    input_ids = batch['input_ids'].to(model_engine.device)
    attention_mask = batch['attention_mask'].to(model_engine.device)
    labels = batch['labels'].to(model_engine.device)
    
    # 前向传播
    outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    
    return {
        'loss': loss
    }

def load_global_tokens(audio_tokenizer, directory, device):
    """加载参考音频的全局token"""
    ref_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                ref_files.append(Path(os.path.join(root, file)))
   
    logger.info(f'Loading global tokens from {ref_files}')
    all_global_tokens = []
    all_global_tokens_ids = []
    characters = []
    for ref_file in ref_files:
        parent_dir = os.path.dirname(ref_file)
        global_tokens_ids, semantic_tokens = audio_tokenizer.tokenize(ref_file)
        global_tokens_ids = global_tokens_ids.squeeze(0).squeeze(0).detach().cpu().tolist()
        global_tokens = "".join([f"<|bicodec_global_{token}|>" for token in global_tokens_ids])
        all_global_tokens.append(global_tokens)
        all_global_tokens_ids.append(global_tokens_ids)
        characters.append(parent_dir.split('/')[-1])
    return all_global_tokens, all_global_tokens_ids, characters

def generate_demo(model_engine, audio_tokenizer, text_tokenizer, demo_text, global_tokens_str, global_tokens_ids, device, eos_token_id, output_dir, epoch, step, character):
    """生成demo音频"""
    # 将模型设置为评估模式
    model_engine.eval()
    
    # 根据 DeepSpeed stage 处理模型
    # 在 stage3 中，我们需要使用 model_engine 而不是 model_engine.module
    # 因为参数分布在不同的进程中
    model = model_engine
    
    # 准备输入文本
    input_text = generate_tts_text(demo_text, global_tokens_str)
    print(f'input_text: {input_text}')
    
    # 准备模型输入
    model_inputs = text_tokenizer(input_text, return_tensors="pt").to(device)
    print(model_inputs)
    len_of_input = model_inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        # 使用 model_engine 进行生成，它会自动处理分布式参数
        generated_tokens = model_engine.generate(
            **model_inputs,
            max_length=2048,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            eos_token_id=eos_token_id,
        )[0]
    
    generated_tokens = generated_tokens[len_of_input:]
    print(f'generated_tokens: {generated_tokens}')
    predicts = text_tokenizer.decode(generated_tokens)
    print(f'predicts: {predicts}')
    predicts = re.findall(r"<|bicodec_semantic_(\d+)|>", predicts)
    print(f'predicts: {predicts}')
    pred_semantic_ids = [int(p) for p in predicts if p.isdigit()]
    if len(pred_semantic_ids) == 0:
        print(f'pred_semantic_ids is empty, predicts: {predicts}')
    else:
        pred_semantic_ids = torch.tensor([pred_semantic_ids], dtype=torch.long).to(device)
        print(f'pred_semantic_ids: {pred_semantic_ids}')
        global_tokens_ids = torch.tensor(global_tokens_ids, dtype=torch.long).unsqueeze(0).to(device)
        print(f'global_tokens_ids: {global_tokens_ids.shape}, pred_semantic_ids: {pred_semantic_ids.shape}')
        print(f'global_tokens_ids: {global_tokens_ids}')
        
        with torch.no_grad():
            wav = audio_tokenizer.detokenize(global_tokens_ids, pred_semantic_ids)
        
        # 保存生成的音频
        output_path = os.path.join(output_dir, f"demo_epoch_{epoch}_step_{step}_{character}.wav")
        sf.write(output_path, wav, audio_tokenizer.config['sample_rate'])
        logger.info(f"Generated demo saved to {output_path}")
    
    # 恢复训练模式
    model_engine.train()

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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # 在 rank0 上初始化音频 tokenizer 和加载 demo 数据
    audio_tokenizer = None
    all_global_tokens = None
    all_global_tokens_ids = None
    characters = None
    eos_token_id = None
    
    if is_main_process and args.demo_dir:
        logger.info(f"Initializing audio tokenizer and loading demo data from {args.demo_dir}")
        audio_tokenizer = BiCodecTokenizer(args.spark_model_dir, device=device)
        all_global_tokens, all_global_tokens_ids, characters = load_global_tokens(audio_tokenizer, args.demo_dir, device)
        eos_token_id = tokenizer.encode("<sementic_end>")[0]
        logger.info(f"Loaded {len(characters)} demo characters")
    
    # Load dataset
    if is_main_process:
        logger.info(f"Loading dataset from {args.data_file}")
    dataset = load_spark_jsonl_dataset(args.data_file)
    dataset = dataset.map(convert_to_tts_format,
                          num_proc=4,  # 使用4个进程
                          remove_columns=dataset['train'].column_names,  # 删除所有原有特征
                          desc="Converting to TTS format"  # 显示进度条描述
                          )
    print(dataset['train'][0])
    # Setup data loading
    if is_main_process:
        logger.info(f"Creating DataLoader with batch size {args.per_device_train_batch_size}, world size {world_size}")
    sampler = DistributedSampler(
        dataset['train'],
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        seed=args.seed
    )
    sementic_start_id = tokenizer.encode("<sementic_start>")[0]
    data_collator = partial(collate_fn, tokenizer=tokenizer, max_length=args.max_length, pad_to_max_length=False, semantic_start_id=sementic_start_id)
    dataloader = DataLoader(
        dataset['train'],
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
                    "overlap_comm": True,
                    "contiguous_gradients": True
                },
                "zero_force_ds_cpu_initialization": True,
                "gradient_checkpointing": args.gradient_checkpointing,
                "dump_state": True
            }
        
    # Init model with deepspeed
    if is_main_process:
        logger.info(f"Initializing model with DeepSpeed config")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    
    if args.ckpt_file is not None:
        if is_main_process:
            logger.info(f"Loading checkpoint from {args.ckpt_file}")
        info = model.load_state_dict(torch.load(args.ckpt_file))
        if is_main_process:
            logger.info(f"Loaded checkpoint info: {info}")
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
    all_tokens = 0
    global_steps = 0
    
    for epoch in range(args.num_epochs):
        if is_main_process:
            update_time = time.time()
            logger.info(f"Epoch {epoch} starts training")
        # 使用时间戳生成随机种子
        time_seed = int(time.time() * 1000) & 0xffffffff  # 获取毫秒级时间戳并转换为32位整数
        sampler.set_epoch(time_seed)  # 使用时间戳作为种子
        
        for batch_idx, batch in enumerate(dataloader):
            if is_main_process:
                input_ids_shape = batch['input_ids'].shape
                attention_mask_shape = batch['attention_mask'].shape
                labels_shape = batch['labels'].shape
                logger.debug(f'input_ids_shape: {input_ids_shape} attention_mask_shape: {attention_mask_shape} labels_shape: {labels_shape} at batch_idx: {batch_idx}')
            
            output = train_step(model_engine, batch)
            loss = output['loss']
            global_steps += 1
            
            # 生成 demo
            if is_main_process and args.demo_dir and global_steps % args.demo_every_steps == 0 and global_steps > 0:
                logger.info(f"Generating demo at epoch {epoch}, step {batch_idx}")
                print(f'batch["input_ids"]: {batch["input_ids"]} shape: {batch["input_ids"].shape}')
                print(f'batch["attention_mask"]: {batch["attention_mask"]} shape: {batch["attention_mask"].shape}')
                print(f'batch["labels"]: {batch["labels"]} shape: {batch["labels"].shape}')
                print(f'eos_token_id: {eos_token_id},characters: {characters},demo_text: {demo_text},device: {device},output_dir: {args.output_dir},epoch: {epoch},batch_idx: {batch_idx}')
                for global_tokens_str, global_tokens_ids, character in zip(all_global_tokens, all_global_tokens_ids, characters):
                    generate_demo(
                        model_engine,
                        audio_tokenizer,
                        tokenizer,
                        demo_text,
                        global_tokens_str,
                        global_tokens_ids,
                        device,
                        eos_token_id,
                        args.demo_dir,
                        epoch,
                        batch_idx,
                        character
                    )
            
            # 首先检测 NaN
            is_nan_loss = torch.isnan(loss) or torch.isinf(loss)
            # 确保所有进程获得相同的 is_nan_loss 值
            is_nan_loss_tensor = torch.tensor([1.0 if is_nan_loss else 0.0], device=model_engine.device)
            # 所有进程同步这个张量以获取一致决策
            torch.distributed.all_reduce(is_nan_loss_tensor, op=torch.distributed.ReduceOp.MAX)
            is_nan_loss = bool(is_nan_loss_tensor.item())

            if is_nan_loss:
                # 使用一个安全的替代 loss 进行 backward
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
                    save_checkpoint(model_engine, args.output_dir, epoch, batch_idx, logger)
            
            # 累计统计
            if is_main_process:
                elapsed_time = time.time()-update_time
                total_loss += loss.item()
                total_steps += 1
                
                # 计算平均值
                avg_loss = total_loss / total_steps
                tokens = batch['input_ids'].numel() * world_size
                all_tokens += tokens
                kts = tokens / elapsed_time / 1e3
                
                # 记录到wandb
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({
                    "loss": loss.item(),
                    "avg_loss": avg_loss,
                    "epoch": epoch,
                    "step": total_steps,
                    "KT/s": kts,
                    "Gtokens": all_tokens/1e9,
                    "learning_rate": current_lr
                })
                
                pbar.update(1)
                pbar.set_postfix({
                    'loss': loss.item(),
                    'avg_loss': avg_loss,
                    'lr': current_lr
                })
        
        #save checkpoint at the end of each epoch
        if args.ds_stage == 3 or args.ds_stage == 2:
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