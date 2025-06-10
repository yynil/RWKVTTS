from ast import mod
from calendar import c
import os
from turtle import up
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
from functools import partial
import time  
import regex as re
from data.spark.multiple_webdataset import MultipleWebDataset, collate_fn_with_tokenizer, process_single_batch_with_audio_tokenizer,process_single_batch_with_audio_tokenizer_culens
from sparktts.models.audio_tokenizer import BiCodecTokenizer
import soundfile as sf
from pathlib import Path
import numpy as np
from inference.rwkv7speech_inference import create_inputs

logger = logging.getLogger(__name__)
demo_text = "我们都是来自同一个地球，同一个梦想！同一片蓝天下！"
@dataclass
class ScriptArguments:
    """Command line arguments for training script"""
    data_file: str = field(
        default=None,
        metadata={"help": "Path to training data file (JSONL format)"}
    )
    webdataset_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to webdataset directory. If set, will ignore data_file and process audio directly"}
    )
    from_index: int = field(
        default=0,
        metadata={"help": "Start index for tar files (inclusive)"}
    )
    to_index: int = field(
        default=1000,
        metadata={"help": "End index for tar files (exclusive)"}
    )
    audio_model_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to audio tokenizer model directory. Required if webdataset_dir is set"}
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

    use_culens: bool = field(
        default=False,
        metadata={"help": "Use cu_seqlens"}
    )

    max_cu_seqlens: int = field(
        default=8192,
        metadata={"help": "Max cu_seqlens"}
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

def update_learning_rate(optimizer, current_step, total_steps, warmup_steps, learning_rate, learning_rate_final,args,is_main_process):
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

def train_step(model_engine, input_embs,labels,cu_seqlens=None,attention_mask=None):
    """执行一步训练"""
    
    # 前向传播
    if cu_seqlens is not None:
        outputs = model_engine(inputs_embeds=input_embs, cu_seqlens=cu_seqlens, labels=labels,use_cache=False)
    else:
        outputs = model_engine(inputs_embeds=input_embs, attention_mask=attention_mask, labels=labels,use_cache=False)
    loss = outputs.loss
    if torch.isnan(loss) or torch.isinf(loss):
        print(f'loss is nan or inf, loss: {loss}')
        print(f'outputs: {outputs}')
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

def generate_demo(model_engine, text_tokenizer, demo_text, global_tokens_ids, device, eos_token_id, output_dir, epoch, step, character, audio_tokenizer):
    """生成demo音频"""
    # 将模型设置为评估模式
    model_engine.eval()
    
    # 准备输入文本
    print(f'input_text: {demo_text}')
    print(f'character: {character},device: {device},eos_token_id: {eos_token_id}')
    
    # 准备模型输入
    input_ids_embs, attention_mask = create_inputs([demo_text], [global_tokens_ids], [[]], text_tokenizer, model_engine)
    
    with torch.no_grad():
        # 使用 model_engine 进行生成
        generated_tokens = model_engine.generate(
            inputs_embeds=input_ids_embs,
            attention_mask=attention_mask,
            max_length=2048,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            eos_token_id=eos_token_id,
        )[0]
    
    print(f'generated_tokens: {generated_tokens}')
    pred_semantic_ids = generated_tokens.tolist()
    if len(pred_semantic_ids) == 0:
        print(f'pred_semantic_ids is empty, pred_semantic_ids: {pred_semantic_ids}')
    else:
        print(f'pred_semantic_ids: {pred_semantic_ids}')
        
        json_output_data = {
            "global_tokens_ids": global_tokens_ids,
            "pred_semantic_ids": pred_semantic_ids,
            "text": demo_text
        }
        output_file = os.path.join(output_dir, f"demo_epoch_{epoch}_step_{step}_{character}.json")
        with open(output_file, "w") as f:
            json.dump(json_output_data, f)
        
        print(f"Generated demo saved to {output_file}")
        print(f"Reconstructing demo...")
        import soundfile as sf
        try:
            with torch.no_grad():
                wav_file = os.path.join(output_dir, f"demo_epoch_{epoch}_step_{step}_{character}.wav")
                global_tokens = torch.tensor(global_tokens_ids).unsqueeze(0).to(device)
                semantic_tokens = torch.tensor(pred_semantic_ids).unsqueeze(0).to(device)
                wav_reconstructed = audio_tokenizer.detokenize(global_tokens, semantic_tokens)
                sf.write(wav_file, wav_reconstructed, 16000)
                print(f"Reconstructed demo saved to {wav_file}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error generating demo: {e}")
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
    
    # 初始化音频 tokenizer
    if args.audio_model_dir is None:
        raise ValueError("audio_model_dir must be set")
    if is_main_process:
        logger.info(f"Initializing audio tokenizer from {args.audio_model_dir}")
    
    
    # 在 rank0 上初始化音频 tokenizer 和加载 demo 数据
    all_global_tokens = None
    all_global_tokens_ids = None
    characters = None
    eos_token_id = None
    
    if is_main_process and args.demo_dir:
        logger.info(f"Initializing audio tokenizer and loading demo data from {args.demo_dir}")
        demo_audio_tokenizer = BiCodecTokenizer(args.spark_model_dir, device='cpu')
        all_global_tokens, all_global_tokens_ids, characters = load_global_tokens(demo_audio_tokenizer, args.demo_dir, device)
        logger.info(f"Loaded {len(characters)} demo characters")
        del demo_audio_tokenizer
        torch.cuda.empty_cache() 
    
    # Load dataset
    if args.webdataset_dir is not None:
        logger.info(f"Loading dataset from webdataset directory: {args.webdataset_dir}")
        dataset = MultipleWebDataset(
            data_dir=args.webdataset_dir,
            target_sr=16000,
            target_channels=1,
            shuffle=False,
            verify_tar=False
        )
    else:
        raise ValueError("webdataset_dir must be set")
    
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
                    "overlap_comm": False,
                    "contiguous_gradients": True
                },
                "zero_force_ds_cpu_initialization": True,
                "gradient_checkpointing": args.gradient_checkpointing,
                "dump_state": False
            }
        
    # Init model with deepspeed
    if is_main_process:
        logger.info(f"Initializing model with DeepSpeed config")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    eos_token_id = model.config.vocab_size - 1
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
    model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config,
            model_parameters=model.parameters(),
            optimizer=optimizer
    )
    
    # 在deepspeed初始化完成后删除原始model
    del model

    # 使用新的 collate 函数
    data_collator = partial(collate_fn_with_tokenizer,  
                            tokenizer=tokenizer)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        num_workers=8,  # 增加工作进程数
        pin_memory=True,
        drop_last=True,
        collate_fn=data_collator,
        persistent_workers=True  # 保持工作进程存活
    )
        
    if is_main_process:
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
    #We do drop out at the training loop which is a lack of feature in the model
    dropout = torch.nn.Dropout(0.02)

    audio_tokenizer = BiCodecTokenizer(args.audio_model_dir, device=model_engine.device)
    for epoch in range(args.num_epochs):
        model_engine.train()
        if is_main_process:
            update_time = time.time()
            logger.info(f"Epoch {epoch} starts training")
            from tqdm import tqdm
            pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")
        # 使用时间戳生成随机种子
        time_seed = int(time.time() * 1000) & 0xffffffff  # 获取毫秒级时间戳并转换为32位整数
        sampler.set_epoch(time_seed)  # 使用时间戳作为种子
        
        for batch_idx, batch in enumerate(dataloader):
            # 更新学习率
            update_learning_rate(
                optimizer,
                global_steps,
                total_steps,
                args.warmup_steps,
                args.learning_rate,
                args.learning_rate_final,
                args,
                is_main_process
            )
            
            # 使用 process_single_batch 处理数据
            if args.use_culens:
                processed_batch = process_single_batch_with_audio_tokenizer_culens(batch, model_engine, audio_tokenizer, eos_token_id)
            else:
                processed_batch = process_single_batch_with_audio_tokenizer(batch, model_engine, audio_tokenizer, eos_token_id)
                maxium_tokens = args.max_tokens_k * 1024  # 将 K 转换为实际 token 数
                current_batch_size,current_batch_seq_len,_ = processed_batch["input_embs"].shape
                max_batch_size = maxium_tokens // current_batch_seq_len
                if max_batch_size < current_batch_size:
                    print(f'max_batch_size < current_batch_size, max_batch_size: {max_batch_size}, current_batch_size: {current_batch_size} shrink the batch size')
                    processed_batch["input_embs"] = processed_batch["input_embs"][:max_batch_size]
                    processed_batch["labels"] = processed_batch["labels"][:max_batch_size]
                    processed_batch["attention_mask"] = processed_batch["attention_mask"][:max_batch_size]
            if is_main_process and batch_idx == 0:
                print(f'input_embs shape: {processed_batch["input_embs"].shape}')
                print(f'labels shape: {processed_batch["labels"].shape}')
                if args.use_culens:
                    print(f'cu_seqlens shape: {processed_batch["cu_seqlens"].shape}')
                    print(f'cu_seqlens: {processed_batch["cu_seqlens"]}')
                else:
                    print(f'attention_mask shape: {processed_batch["attention_mask"].shape}')
                    print(f'attention_mask: {processed_batch["attention_mask"]}')
            output = train_step(model_engine, **processed_batch)
            loss = output['loss']
            if is_main_process and batch_idx == 0:
                print(f'loss: {loss}')
            global_steps += 1
            
            # 生成 demo
            if is_main_process and args.demo_dir and global_steps % args.demo_every_steps == 0 and global_steps > 0:
                logger.info(f"Generating demo at epoch {epoch}, step {batch_idx}")
                for global_tokens_str, global_tokens_ids, character in zip(all_global_tokens, all_global_tokens_ids, characters):
                    generate_demo(
                        model_engine,
                        tokenizer,
                        demo_text,
                        global_tokens_ids,
                        device,
                        eos_token_id,
                        args.demo_dir,
                        epoch,
                        batch_idx,
                        character,
                        audio_tokenizer
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
                # safe_loss = loss * 0.0
                # print(f'safe_loss: {safe_loss}')
                if is_main_process:
                    logger.warning(f"NaN loss detected at batch {batch_idx}, using safe zero loss instead")
                    wandb.log({
                        "nan_detected": 1,
                        "epoch": epoch,
                        "step": total_steps
                    })
                
                # 所有节点都执行 backward，但使用的是零梯度
                # model_engine.backward(safe_loss)
                # model_engine.step()  # 这步实际上不会改变参数，因为梯度是零
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
                tokens = processed_batch["input_embs"].shape[1] * world_size
                all_tokens += tokens
                kts = tokens / elapsed_time / 1e3
                
                # 记录到wandb
                current_lr = optimizer.param_groups[0]['lr']
                log_metrics(optimizer, loss, avg_loss, epoch, total_steps, kts, all_tokens, current_lr)
                
                pbar.update(1)
                pbar.set_postfix({
                    'loss': loss.item(),
                    'avg_loss': avg_loss,
                    'lr': current_lr
                })
        
        #save checkpoint at the end of each epoch
        if args.ds_stage == 3 or args.ds_stage == 2:
            epoch_checkpoint_dir = f"{args.output_dir}/epoch_{epoch}"
            os.makedirs(epoch_checkpoint_dir,exist_ok=True)
            print(f'saving checkpoint to {epoch_checkpoint_dir}')
            model_engine.save_checkpoint(epoch_checkpoint_dir)
        
        if is_main_process:
            pbar.close()
    
    # 训练结束后关闭wandb
    if is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main() 