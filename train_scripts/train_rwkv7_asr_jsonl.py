#!/usr/bin/env python3
"""
基于RWKV7ASRModel的ASR训练脚本
"""

import os
import sys
import json
import logging
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
import deepspeed
import wandb
from tqdm import tqdm
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.llm.rwkv_asr import RWKV7ASRModel
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Config, RWKV7Model, RWKV7ForCausalLM
from tokenizer.rwkv_tokenizer import RWKV_TOKENIZER

from rwkvfla.modules.l2warp import l2_warp
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量，用于存储预编码的指令和hints
_global_text_input_ids_chinese = None
_global_text_input_ids_english = None
_global_hints_ids = None

def create_asr_inputs_and_labels(batch, tokenizer, eos_id=0, pad_id=0):
    """为ASR任务创建输入和标签，使用RWKV7ASRModel的格式"""
    from langdetect import detect, LangDetectException
    
    def detect_language(text):
        try:
            return detect(text)
        except LangDetectException:
            return 'en'
    
    # 使用全局预编码变量
    text_input_ids_chinese = _global_text_input_ids_chinese
    text_input_ids_english = _global_text_input_ids_english
    hints_ids = _global_hints_ids
    audio_input_ids_list = []
    text_input_ids_list = []
    audio_attention_mask_list = []
    text_attention_mask_list = []
    labels_list = []
    
    for item in batch:
        semantic_tokens = item['semantic_tokens']
        text = item['text']
        
        # 检测语言并选择对应的指令
        language = detect_language(text)
        if language.startswith('zh'):
            text_input_ids = text_input_ids_chinese.clone()
        else:
            text_input_ids = text_input_ids_english.clone()
        
        # 音频输入：直接使用semantic_tokens
        audio_input_ids = torch.tensor(semantic_tokens, dtype=torch.long)
        audio_attention_mask = torch.ones(len(semantic_tokens), dtype=torch.long)
        
        # 文本输入：指令 + 目标文本
        target_text_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        full_text_input_ids = torch.cat([text_input_ids, target_text_ids, torch.tensor([eos_id], dtype=torch.long)], dim=0)
        text_attention_mask = torch.ones(len(full_text_input_ids), dtype=torch.long)
        
        # 标签：只对目标文本部分计算损失
        labels = torch.full((len(full_text_input_ids),), -100, dtype=torch.long)
        start_idx = len(text_input_ids) - 1
        end_idx = start_idx + len(target_text_ids)
        labels[start_idx:end_idx] = target_text_ids
        labels[end_idx] = eos_id
        
        audio_input_ids_list.append(audio_input_ids)
        text_input_ids_list.append(full_text_input_ids)
        audio_attention_mask_list.append(audio_attention_mask)
        text_attention_mask_list.append(text_attention_mask)
        labels_list.append(labels)
    
    # 填充到相同长度（左填充）
    from torch.nn.utils.rnn import pad_sequence
    
    # 对于LLM训练，使用左填充更合适
    # 音频序列也使用左填充（保持顺序）
    audio_input_ids = pad_sequence(audio_input_ids_list, batch_first=True, padding_value=0, padding_side='left')
    audio_attention_mask = pad_sequence(audio_attention_mask_list, batch_first=True, padding_value=0, padding_side='left')
    
    # 文本相关序列使用左填充
    text_input_ids = pad_sequence(text_input_ids_list, batch_first=True, padding_value=0, padding_side='left')
    text_attention_mask = pad_sequence(text_attention_mask_list, batch_first=True, padding_value=0, padding_side='left')
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100, padding_side='left')
    
    return audio_input_ids, text_input_ids, audio_attention_mask, text_attention_mask, labels, hints_ids

def train_step(model_engine, audio_input_ids, text_input_ids, audio_attention_mask, text_attention_mask, labels, hints_ids):
    """训练步骤"""
    output = model_engine(audio_input_ids, text_input_ids, audio_attention_mask, text_attention_mask,labels, hints_ids)
    loss = output.loss
    return loss

def compute_gradient_norm(model_engine):
    """计算梯度范数"""
    total_norm = 0.0
    for p in model_engine.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def configure_optimizer(model, args):
    """配置优化器，使用RWKV7标准方法，只训练audio_lm_model和projector"""
    lr_1x = set()
    lr_2x = set()
    lr_decay = set()    
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        # 只训练audio_lm_model和projector，冻结llm参数
        if "llm." in n:
            p.requires_grad = False
            continue
        
        # 注意力相关参数使用2倍学习率
        if "att.w0" in n:
            lr_2x.add(n)
        # 权重参数使用权重衰减
        elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0) and (".weight" in n):
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
            print(f'param_group: {param_group["name"]} lr: {param_group["lr"]} weight_decay: {param_group["weight_decay"]}, params: {len(param_group["params"])}')

def save_checkpoint(model_engine, output_dir, epoch, step, training_state, logger):
    """Save model checkpoint with training state"""
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
    
    # 保存模型参数和训练状态
    model_engine.save_checkpoint(output_dir, client_state=training_state)
    
    # 额外保存训练状态文件，便于直接加载
    if model_engine.local_rank == 0:
        client_state_file = os.path.join(output_dir, 'client_states.pt')
        torch.save(training_state, client_state_file)
        print(f"额外保存训练状态到: {client_state_file}")

def log_metrics(wandb, global_step, loss, avg_loss, learning_rate, all_tokens, is_main_process):
    """记录训练指标到wandb"""
    if is_main_process:
        wandb.log({
            "step": global_step,
            "loss": loss,
            "avg_loss": avg_loss,
            "learning_rate": learning_rate,
            "Gtokens": all_tokens/(1000*1000*1000),
        })

def get_training_state(model_engine, global_step, epoch, all_tokens):
    """获取训练状态，用于保存到checkpoint"""
    return {
        "global_step": global_step,
        "epoch": epoch,
        "all_tokens": all_tokens,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_model_path", type=str,required=True, help="音频模型目录路径")
    parser.add_argument("--llm_model_path", type=str, required=True, help="LLM模型目录路径")
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录")
    parser.add_argument("--output_dir", type=str, default="outputs/rwkv7_asr", help="输出目录")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="每个设备的训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--learning_rate_final", type=float, default=1e-5, help="最终学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="学习率预热步数")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--ds_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="启用梯度检查点")
    parser.add_argument("--save_steps", type=int, default=1000, help="保存checkpoint的步数")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从checkpoint恢复训练")
    parser.add_argument("--ds_optimizer_offload", action="store_true", help="启用DeepSpeed优化器CPU卸载")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--max_k_tokens_per_batch", type=int, required=True, help="每个batch的最大token数")
    parser.add_argument("--gradient_clipping", type=float, default=0.5, help="梯度裁剪阈值")
    
    args = parser.parse_args()
    
    # 初始化DeepSpeed
    deepspeed.init_distributed()
    
    # 获取分布式训练信息
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    
    # 设置设备
    device = torch.device("cuda", args.local_rank)
    is_main_process = args.local_rank in [-1, 0]
    
    if is_main_process:
        print(f"DeepSpeed初始化完成，world_size={world_size}, local_rank={args.local_rank}")
        print(f"使用设备: {device}")
    
    # # 创建音频模型配置
    # audio_config = {
    #     "a_low_rank_dim": 64,
    #     "attn": None,
    #     "attn_mode": "chunk",
    #     "bos_token_id": 0,
    #     "decay_low_rank_dim": 64,
    #     "eos_token_id": 0,
    #     "fuse_cross_entropy": True,
    #     "fuse_norm": False,
    #     "gate_low_rank_dim": 128,
    #     "head_dim": 64,
    #     "hidden_act": "sqrelu",
    #     "hidden_ratio": 4.0,
    #     "hidden_size": 768,
    #     "initializer_range": 0.006,
    #     "intermediate_size": 3072,
    #     "max_position_embeddings": 2048,
    #     "norm_bias": True,
    #     "norm_eps": 1e-05,
    #     "norm_first": True,
    #     "num_heads": 32,
    #     "num_hidden_layers": 12,
    #     "tie_word_embeddings": False,
    #     "use_cache": True,
    #     "v_low_rank_dim": 32,
    #     "vocab_size": 8192
    # }
    
    # 创建音频模型
    if is_main_process:
        print("创建音频模型...")
    audio_lm_model = RWKV7Model.from_pretrained(args.audio_model_path, trust_remote_code=True)
    if is_main_process:
        print("音频模型创建完成")
    
    # 加载LLM模型
    if is_main_process:
        print(f"加载LLM模型: {args.llm_model_path}")
    llm = RWKV7ForCausalLM.from_pretrained(args.llm_model_path, trust_remote_code=True)
    if is_main_process:
        print("LLM模型加载完成")
    
    # 创建ASR模型
    if is_main_process:
        print("创建ASR模型...")
    model = RWKV7ASRModel(audio_lm_model, llm)
    model.train()
    if is_main_process:
        print("ASR模型创建完成")
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        if is_main_process:
            print(f"加载参数: {args.resume_from_checkpoint}")
        info = model.load_state_dict(torch.load(args.resume_from_checkpoint), strict=False)
        print(f"加载参数信息: {info}")
    if is_main_process:
        print(f"模型创建完成")
        print(f"总参数数量: {sum(1 for param in model.parameters())}")
        print(f"可训练参数数量: {sum(1 for param in model.parameters() if param.requires_grad)}")
    
    # 从checkpoint恢复训练状态
    global_step = 0
    all_tokens = 0
    start_epoch = 0
    
    
    # 加载tokenizer
    vocab_file = os.path.join(args.llm_model_path, "rwkv_vocab_enlarged.txt")
    if not os.path.exists(vocab_file):
        vocab_file = "tokenizer/rwkv_vocab_v20230424.txt"
    
    tokenizer = RWKV_TOKENIZER(vocab_file)
    
    # 初始化全局预编码变量
    if is_main_process:
        print("初始化全局预编码变量...")
    
    # 预定义的中英文指令
    chinese_instruction = "User: 把以上音频转化为中文。"
    english_instruction = "User: Convert the audios to English."
    # 预编码指令
    global _global_text_input_ids_chinese, _global_text_input_ids_english, _global_hints_ids
    _global_text_input_ids_chinese = torch.tensor(tokenizer.encode(chinese_instruction), dtype=torch.long)
    _global_text_input_ids_english = torch.tensor(tokenizer.encode(english_instruction), dtype=torch.long)
    _global_hints_ids = torch.tensor(tokenizer.encode("\nAssistant:"), dtype=torch.long)
    
    if is_main_process:
        print(f"中文指令长度: {len(_global_text_input_ids_chinese)}")
        print(f"英文指令长度: {len(_global_text_input_ids_english)}")
        print(f"hints长度: {len(_global_hints_ids)}")
    
    # 加载数据集
    if is_main_process:
        print(f"加载数据集: {args.data_dir}")
    
    jsonl_files = [str(p) for p in Path(args.data_dir).rglob("*.jsonl")]
    if not jsonl_files:
        raise ValueError(f"No .jsonl files found in {args.data_dir}")
    
    if is_main_process:
        print(f"找到 {len(jsonl_files)} 个JSONL文件")
        print(f"前5个文件: {jsonl_files[:5]}")
    
    dataset = load_dataset("json", data_files=jsonl_files, split="train")
    
    if is_main_process:
        print(f"数据集大小: {len(dataset)}")
        if len(dataset) > 0:
            print(f"数据集示例: {dataset[0]}")
    
    # 创建数据加载器
    def collate_fn(batch):
        return batch
    
    if is_main_process:
        print(f"创建数据加载器，batch_size={args.per_device_train_batch_size}")
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    if is_main_process:
        print(f"数据加载器创建完成，总批次数: {len(train_dataloader)}")
    
    # 配置优化器
    optimizer = configure_optimizer(model, args)
    
    # DeepSpeed配置
    train_batch_size = args.per_device_train_batch_size * world_size
    ds_config = {
        "distributed_backend": "nccl",
        "train_batch_size": train_batch_size,
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": args.ds_stage,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "overlap_comm": False,
            "contiguous_gradients": True
        },
        "gradient_checkpointing": args.gradient_checkpointing,
        "dump_state": False,
        # 添加梯度裁剪配置
        "gradient_clipping": args.gradient_clipping
    }
    
    # 根据stage添加特定配置
    if args.ds_stage >= 2:
        ds_config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
            "buffer_count": 4
        }
    
    if args.ds_stage >= 3:
        ds_config["zero_optimization"].update({
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
            }
        })
        ds_config["zero_force_ds_cpu_initialization"] = True
    
    # 初始化DeepSpeed引擎
    if is_main_process:
        print("初始化DeepSpeed引擎...")
        print(f"DeepSpeed配置: {ds_config}")
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )
    
    if is_main_process:
        print("DeepSpeed引擎初始化完成")
        print(f"梯度裁剪阈值设置为: {args.gradient_clipping}")
    
    model_engine.train()
    
    # Warmup model to compile Triton kernels
    if is_main_process:
        print(f'warming up the model')
    

    
    # 初始化wandb
    if is_main_process:
        wandb.init(project="rwkv7-asr", name="asr-training")
    
    # 训练循环
    if is_main_process:
        print(f"开始训练循环，从epoch {start_epoch}开始，总共{args.num_train_epochs}个epoch")
    
    for epoch in range(start_epoch, args.num_train_epochs):
        if is_main_process:
            print(f"开始epoch {epoch+1}/{args.num_train_epochs}")
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")
        sum_loss = 0
        
        for step, batch in enumerate(progress_bar):
            # 第一个batch的调试信息
            if step == 0 and is_main_process:
                print(f"处理第一个batch，batch大小: {len(batch)}")
                print(f"Batch示例: {batch[0] if batch else 'Empty batch'}")
            
            # 更新学习率
            update_learning_rate(
                optimizer,
                global_step,
                len(train_dataloader) * args.num_train_epochs,
                args.warmup_steps,
                args.learning_rate,
                args.learning_rate_final,
                args,
                is_main_process
            )
            
            # 创建输入和标签
            try:
                audio_input_ids, text_input_ids, audio_attention_mask, text_attention_mask, labels, hints_ids = create_asr_inputs_and_labels(batch, tokenizer)
                input_size_per_sample = (audio_input_ids.shape[1] + text_input_ids.shape[1])
                current_k_tokens = input_size_per_sample* audio_input_ids.shape[0]
                if current_k_tokens > args.max_k_tokens_per_batch * 1024:
                    print(f"当前batch的token数超过最大限制: {current_k_tokens} > {args.max_k_tokens_per_batch}")
                    max_bsz = args.max_k_tokens_per_batch * 1024 // input_size_per_sample
                    print(f"调整batch大小为: {max_bsz}")
                    audio_input_ids = audio_input_ids[:max_bsz, :]
                    text_input_ids = text_input_ids[:max_bsz, :]
                    audio_attention_mask = audio_attention_mask[:max_bsz, :]
                    text_attention_mask = text_attention_mask[:max_bsz, :]
                    labels = labels[:max_bsz, :]
                    # hints_ids不需要调整，因为它是全局共享的
                if step == 0 and is_main_process:
                    print(f"输入形状: audio={audio_input_ids.shape}, text={text_input_ids.shape}, labels={labels.shape}")
                    
            except Exception as e:
                if is_main_process:
                    print(f"数据处理失败: {e}")
                    print(f"问题batch: {batch}")
                continue
            
            # 移动到设备
            audio_input_ids = audio_input_ids.to(device)
            text_input_ids = text_input_ids.to(device)
            audio_attention_mask = audio_attention_mask.to(device)
            text_attention_mask = text_attention_mask.to(device)
            labels = labels.to(device)
            hints_ids = hints_ids.to(device)
            
            # 前向传播和反向传播
            loss = train_step(model_engine, audio_input_ids, text_input_ids, audio_attention_mask, text_attention_mask, labels, hints_ids)
            
            # 检查loss是否为nan - 使用分布式同步
            is_nan_or_inf = torch.isnan(loss) or torch.isinf(loss)
            if torch.distributed.is_initialized():
                # 在所有进程间同步nan状态
                nan_tensor = torch.tensor(1 if is_nan_or_inf else 0, device=device, dtype=torch.long)
                torch.distributed.all_reduce(nan_tensor, op=torch.distributed.ReduceOp.MAX)
                should_skip = nan_tensor.item() > 0
            else:
                should_skip = is_nan_or_inf
            
            if should_skip:
                if is_main_process:
                    print(f"Warning: NaN or Inf loss detected at step {step}, loss = {loss.item()}")
                    print("Skipping this step and resetting gradients...")
                
                # 重置梯度，避免nan传播
                model_engine.zero_grad()
                continue
            
            sum_loss += loss.item()
            avg_loss = sum_loss / (step + 1)
            
            # 反向传播
            model_engine.backward(loss)
            
            # 计算梯度范数并监控
            if step % 100 == 0:
                grad_norm = compute_gradient_norm(model_engine)
                if torch.distributed.is_initialized():
                    # 在所有进程间同步梯度范数
                    grad_norm_tensor = torch.tensor(grad_norm, device=device, dtype=torch.float)
                    torch.distributed.all_reduce(grad_norm_tensor, op=torch.distributed.ReduceOp.MAX)
                    grad_norm = grad_norm_tensor.item()
                
                if is_main_process:
                    print(f"Step {step}: Gradient norm = {grad_norm:.4f}")
                    # 如果梯度范数过大，给出警告
                    if grad_norm > 10.0:
                        print(f"Warning: Large gradient norm detected: {grad_norm:.4f}")
            
            model_engine.step()
            
            # 计算总 tokens（包括音频和文本）
            audio_tokens = audio_input_ids.shape[0] * audio_input_ids.shape[1]
            text_tokens = text_input_ids.shape[0] * text_input_ids.shape[1]
            all_tokens += audio_tokens + text_tokens
            
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录指标（每一步都记录）
            log_metrics(wandb, global_step, loss.item(), avg_loss, current_lr, all_tokens, is_main_process)
            
            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "avg_loss": f"{avg_loss:.4f}",
                "lr": f"{current_lr:.2e}",
                "grad_clip": f"{args.gradient_clipping}"
            })
            
            # 保存checkpoint
            if global_step > 0 and global_step % args.save_steps == 0:
                training_state = get_training_state(model_engine, global_step, epoch, all_tokens)
                save_checkpoint(model_engine, args.output_dir, epoch, global_step, training_state, logger)
            
            global_step += 1
    
    if is_main_process:
        logger.info("训练完成！")

if __name__ == "__main__":
    main()
