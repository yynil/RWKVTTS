#!/usr/bin/env python3
"""
基于RWKV7S2S_SingleFFN的ASR训练脚本 - 简化版本，去掉KL散度，使用位置权重
"""

import os
from pathlib import Path
import sys
import json
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
import wandb
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional, Tuple

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.llm.rwkv_s2s_single_ffn import RWKV7S2S_SingleFFN,L2Wrap
from tokenizer.rwkv_tokenizer import RWKV_TOKENIZER

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScriptArguments:
    """训练脚本的参数配置"""
    
    def __init__(self, model_path=None):
        self.output_dir = "outputs/rwkv7s2s_single_ffn_asr"
        self.num_train_epochs = 5  # 增加训练轮数
        self.per_device_train_batch_size = 8  # 增加批次大小
        self.gradient_accumulation_steps = 2  # 减少梯度累积
        self.learning_rate = 5e-5  # 调整学习率
        self.weight_decay = 0.01
        self.warmup_steps = 1000
        self.logging_steps = 10
        self.save_steps = 500
        self.eval_steps = 500
        self.save_total_limit = 3
        self.dataloader_pin_memory = False
        self.seed = 42
        
        # 模型配置
        self.n_layer = 24
        self.n_embd = 1024
        self.vocab_size = 65536
        self.text_vocab_size = 65536
        self.audio_vocab_size = 8192
        self.head_size_a = 64
        self.head_size_divisor = 1
        self.dropout = 0.1
        self.need_init_tmix = True
        self.need_init_cmix = True
        self.grad_cp = 1
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.local_rank = -1
        
        # 数据配置
        self.data_dir = "data"
        self.max_seq_length = 2048
        
        # DeepSpeed配置
        self.deepspeed_config = None
        self.ds_stage = 2
        self.gradient_checkpointing = True
        
        # 模型路径
        self.model_path = model_path
        
        # 如果提供了模型路径，自动加载配置
        if model_path:
            self.load_model_config(model_path)
    
    def load_model_config(self, model_path):
        """从模型目录加载配置"""
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # 更新模型配置
            self.n_layer = config.get('n_layer', self.n_layer)
            self.n_embd = config.get('n_embd', self.n_embd)
            self.vocab_size = config.get('vocab_size', self.vocab_size)
            self.text_vocab_size = config.get('text_vocab_size', self.text_vocab_size)
            self.audio_vocab_size = config.get('audio_vocab_size', self.audio_vocab_size)
            self.head_size_a = config.get('head_size_a', self.head_size_a)
            self.head_size_divisor = config.get('head_size_divisor', self.head_size_divisor)
            self.dropout = config.get('dropout', self.dropout)
            self.grad_cp = config.get('grad_cp', self.grad_cp)
            
            logger.info(f"从配置文件加载模型参数: {config}")
        else:
            logger.warning(f"配置文件不存在: {config_file}")

def configure_optimizer(model, args):
    """配置优化器，使用不同的学习率"""
    lr_1x = set()
    lr_2x = set()
    lr_decay = set()    
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
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

def save_checkpoint(model_engine, epoch, step, args, loss, avg_loss, is_final_step=False):
    """保存检查点"""
    if os.path.exists(args.output_dir):
        if model_engine.local_rank == 0:
            checkpoints = os.listdir(args.output_dir)
            checkpoints = [f for f in checkpoints if os.path.isdir(os.path.join(args.output_dir, f))]
            checkpoints.sort(key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
            checkpoints = [f for f in checkpoints if not f.startswith("checkpoint-final-")]
            if len(checkpoints) > 2:
                print(f'deleting older checkpoints {checkpoints[0]}')
                import shutil
                shutil.rmtree(os.path.join(args.output_dir, checkpoints[0]))    
    if is_final_step:
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-final-{epoch}")
    else:
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model_engine.save_checkpoint(checkpoint_dir)
    
    # 保存训练状态
    training_state = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "avg_loss": avg_loss
    }
    
    with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
        json.dump(training_state, f)

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

def log_metrics(step, loss, learning_rate, args, avg_loss, all_tokens):
    """记录训练指标"""
    if args.local_rank in [-1, 0]:  # 只在主进程记录
        wandb.log({
            "step": step,
            "loss": loss,
            "learning_rate": learning_rate,
            "avg_loss": avg_loss,
            "Gtokens": all_tokens/(1000*1000*1000),
        })



def compute_positional_cumulative_loss(logits, labels):
    """计算基于位置的累积损失：前面的错误会惩罚后面的所有位置"""
    batch_size, seq_len, vocab_size = logits.shape
    
    # 重塑为2D
    logits_flat = logits.reshape(-1, vocab_size)  # (B*T, vocab_size)
    labels_flat = labels.reshape(-1)  # (B*T)
    
    # 计算每个位置的交叉熵损失
    ce_loss = torch.nn.functional.cross_entropy(logits_flat, labels_flat, reduction='none')  # (B*T)
    
    # 重塑回原始形状
    ce_loss = ce_loss.reshape(batch_size, seq_len)  # (B, T)
    labels_reshaped = labels_flat.reshape(batch_size, seq_len)  # (B, T)
    
    # 创建有效位置mask
    valid_mask = (labels_reshaped != -100)  # (B, T)
    
    # 计算随机猜测的交叉熵损失作为clamp的上限
    random_guess_ce = -torch.log(torch.tensor(1.0 / vocab_size, device=logits.device))
    clamp_value = random_guess_ce * 2.0  # 设置为随机猜测损失的2倍作为上限
    
    # 添加调试信息
    if hasattr(compute_positional_cumulative_loss, 'step_count'):
        compute_positional_cumulative_loss.step_count += 1
    else:
        compute_positional_cumulative_loss.step_count = 0
    
    if compute_positional_cumulative_loss.step_count % 100 == 0:
        print(f"Clamp value: {clamp_value:.4f}, Random guess CE: {random_guess_ce:.4f}")
    
    # 统计每个batch第一个位置的CE值
    first_pos_ce_values = []
    
    # 使用向量化操作计算累积损失，避免循环中的in-place操作
    cumulative_losses = []
    
    for b in range(batch_size):
        # 找到这个batch中有效的位置
        valid_positions = valid_mask[b].nonzero(as_tuple=True)[0]
        
        if len(valid_positions) > 0:
            # 记录第一个位置的CE值
            first_pos_ce = ce_loss[b, valid_positions[0]]
            first_pos_ce_values.append(first_pos_ce.item())
            
            # 提取这个batch的有效损失
            batch_ce_losses = ce_loss[b, valid_positions]  # (N_valid,)
            
            # 计算累积损失：使用cumsum
            # 对于位置i，累积损失 = sum(ce_loss[j] for j <= i)
            cumulative_batch_losses = torch.cumsum(batch_ce_losses, dim=0)  # (N_valid,)
            
            # 应用clamp
            cumulative_batch_losses = torch.clamp(cumulative_batch_losses, max=clamp_value)
            
            cumulative_losses.append(cumulative_batch_losses)
    
    # 合并所有batch的累积损失
    if cumulative_losses:
        all_cumulative_losses = torch.cat(cumulative_losses, dim=0)
        final_loss = all_cumulative_losses.mean()
    else:
        final_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    # 计算第一个位置CE值的统计信息
    if first_pos_ce_values:
        first_pos_ce_mean = sum(first_pos_ce_values) / len(first_pos_ce_values)
        first_pos_ce_min = min(first_pos_ce_values)
        first_pos_ce_max = max(first_pos_ce_values)
        
        # 将统计信息存储到函数属性中，供train_step使用
        compute_positional_cumulative_loss.first_pos_stats = {
            'mean': first_pos_ce_mean,
            'min': first_pos_ce_min,
            'max': first_pos_ce_max,
            'count': len(first_pos_ce_values)
        }
    
    # 检查最终损失
    if torch.isnan(final_loss) or torch.isinf(final_loss):
        print(f"Warning: Final loss is {final_loss}, using safe fallback")
        # 使用简单的平均交叉熵作为fallback
        safe_loss = ce_loss[valid_mask].mean()
        return safe_loss
    
    return final_loss

def train_step(model_engine, input_ids, labels, attention_mask):
    """简化的训练步骤 - 使用累积损失强制模型关注前面tokens"""
    # 直接前向传播
    logits, _ = model_engine(input_ids, attention_mask, is_text=True)
    
    # 检查logits是否有NaN
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("Warning: NaN/Inf detected in logits, returning safe loss values")
        safe_loss = torch.where(torch.isnan(logits) | torch.isinf(logits), 
                               torch.zeros_like(logits), logits).sum() * 0.0
        return safe_loss
    
    # 计算累积损失：前面的错误会惩罚后面的所有位置
    loss = compute_positional_cumulative_loss(logits, labels)
    
    # 添加调试信息
    if hasattr(train_step, 'step_count'):
        train_step.step_count += 1
    else:
        train_step.step_count = 0
    
    if train_step.step_count % 100 == 0:
        print(f"Step {train_step.step_count}: Cumulative Loss: {loss.item():.4f}")
        # 计算常规交叉熵损失作为对比
        with torch.no_grad():
            ce_loss_flat = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none')
            valid_mask = (labels.view(-1) != -100)
            if valid_mask.sum() > 0:
                regular_ce = ce_loss_flat[valid_mask].mean()
                print(f"  Regular CE Loss: {regular_ce.item():.4f}")
            
            # 显示第一个位置的CE统计信息
            if hasattr(compute_positional_cumulative_loss, 'first_pos_stats'):
                stats = compute_positional_cumulative_loss.first_pos_stats
                print(f"  First Position CE - Mean: {stats['mean']:.4f}, Min: {stats['min']:.4f}, Max: {stats['max']:.4f} (from {stats['count']} samples)")
                
                # 计算随机猜测的CE作为参考
                vocab_size = logits.size(-1)
                random_guess_ce = -torch.log(torch.tensor(1.0 / vocab_size, device=logits.device))
                print(f"  Random Guess CE: {random_guess_ce.item():.4f}")
                
                # 如果第一个位置的CE接近随机猜测，说明模型还没有学会从音频预测
                if stats['mean'] > random_guess_ce.item() * 0.9:
                    print(f"  ⚠️  Warning: First position CE is close to random guess, model may not be learning from audio!")
                elif stats['mean'] < random_guess_ce.item() * 0.5:
                    print(f"  ✅ Good: First position CE is significantly better than random guess!")
    
    # 应用L2Wrap
    loss = L2Wrap.apply(loss, logits)
    
    return loss

def create_asr_inputs_and_labels_simple(batch, tokenizer, eos_token_id, pad_token_id):
    """为ASR任务创建简化的输入和标签，支持left shift预测，序列长度必须是16的倍数"""
    from utils.s2s_utilities import create_asr_inputs_and_labels
    
    # 调用utils中的函数（只返回3个值）
    input_ids, labels, attention_mask = create_asr_inputs_and_labels(batch, tokenizer, eos_token_id, pad_token_id)
    
    # 确保序列长度是16的倍数，如果不是则进行左填充
    batch_size, seq_len = input_ids.shape
    target_len = ((seq_len + 15) // 16) * 16
    
    if target_len > seq_len:
        pad_len = target_len - seq_len
        if not hasattr(create_asr_inputs_and_labels_simple, '_printed_padding_warning'):
            print(f"序列长度 {seq_len} 不是16的倍数，左填充 {pad_len} 个token到 {target_len}")
            create_asr_inputs_and_labels_simple._printed_padding_warning = True
        
        new_input_ids = torch.full((batch_size, target_len), pad_token_id, dtype=torch.long)
        new_labels = torch.full((batch_size, target_len), -100, dtype=torch.long)
        new_attention_mask = torch.zeros((batch_size, target_len), dtype=torch.long)
        
        new_input_ids[:, pad_len:] = input_ids
        new_labels[:, pad_len:] = labels
        new_attention_mask[:, pad_len:] = attention_mask
        
        input_ids = new_input_ids
        labels = new_labels
        attention_mask = new_attention_mask
    
    # 对labels进行left shift，使模型能够预测下一个token
    batch_size, seq_len = labels.shape
    shifted_labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    shifted_labels[:, :-1] = labels[:, 1:]
    
    return input_ids, shifted_labels, attention_mask

def load_and_prepare_dataset(args):
    """加载和准备数据集"""
    if args.data_dir is None:
        raise ValueError("data_dir must be set")
    
    logger.info(f"Loading dataset from JSONL files in: {args.data_dir}")
    jsonl_files = [str(p) for p in Path(args.data_dir).rglob("*.jsonl")]
    if not jsonl_files:
        raise ValueError(f"No .jsonl files found in {args.data_dir}")
    
    dataset = load_dataset("json", data_files=jsonl_files, split="train")
    
    return dataset

def collate_fn(batch):
    """数据整理函数"""
    return batch

def main():
    """主训练函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="模型目录路径")
    parser.add_argument("--deepspeed_config", type=str, default=None, help="DeepSpeed配置文件路径（可选）")
    parser.add_argument("--ds_stage", type=int, default=2, help="DeepSpeed ZeRO stage (默认: 2)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="启用梯度检查点")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="每个设备的训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--output_dir", type=str, default="outputs/rwkv7s2s_single_ffn_asr", help="输出目录")
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="权重衰减")
    parser.add_argument("--ds_optimizer_offload", action="store_true", help="启用DeepSpeed优化器CPU卸载")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=1000, help="保存检查点步数")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="学习率预热步数")
    parser.add_argument("--learning_rate_final", type=float, default=1e-5, help="最终学习率")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--max_k_tokens_per_batch", type=int, default=64, help="每个批次的最大序列长度")

    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")

    args = parser.parse_args()
    
    # 初始化DeepSpeed
    deepspeed.init_distributed()
    
    # 设置设备
    device = torch.device("cuda", args.local_rank)
    
    # 获取分布式训练信息
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    is_main_process = args.local_rank in [-1, 0]
    
    # 自动生成DeepSpeed配置
    if args.deepspeed_config:
        if is_main_process:
            logger.info(f"Loading DeepSpeed config from {args.deepspeed_config}")
        with open(args.deepspeed_config, 'r') as f:
            ds_config = json.load(f)
    else:
        # 自动生成DeepSpeed配置
        if is_main_process:
            logger.info("Using auto-generated DeepSpeed config")
        model_args = ScriptArguments(args.model_path)
        # 更新命令行参数
        model_args.per_device_train_batch_size = args.per_device_train_batch_size
        model_args.gradient_accumulation_steps = args.gradient_accumulation_steps
        model_args.learning_rate = args.learning_rate
        model_args.num_train_epochs = args.num_train_epochs
        model_args.output_dir = args.output_dir
        model_args.data_dir = args.data_dir
        model_args.ds_stage = args.ds_stage
        model_args.gradient_checkpointing = args.gradient_checkpointing
        model_args.weight_decay = args.weight_decay
        model_args.ds_optimizer_offload = args.ds_optimizer_offload
        model_args.logging_steps = args.logging_steps
        model_args.save_steps = args.save_steps
        model_args.warmup_steps = args.warmup_steps
        model_args.learning_rate_final = args.learning_rate_final
        model_args.local_rank = args.local_rank
        model_args.grad_cp = 1 if args.gradient_checkpointing else 0
        model_args.dropout = args.dropout
        
        train_batch_size = model_args.per_device_train_batch_size * world_size
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
                "contiguous_gradients": True,
                "round_robin_gradients": True
            },
            "zero_force_ds_cpu_initialization": True,
            "gradient_checkpointing": args.gradient_checkpointing,
            "dump_state": False,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 100
        }
    
    # 确保grad_cp与gradient_checkpointing保持一致
    model_args.grad_cp = 1 if args.gradient_checkpointing else 0
    
    model = RWKV7S2S_SingleFFN(model_args)
    model.train()
    if is_main_process:
        print(f"model: {model}")
    
    # 加载模型权重
    model_files = [f for f in os.listdir(args.model_path) if f.endswith('.pth')]
    if model_files:
        model_file = os.path.join(args.model_path, model_files[0])
        logger.info(f"加载模型权重: {model_file}")
        state_dict = torch.load(model_file, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Missing keys: {missing_keys}")
        logger.info(f"Unexpected keys: {unexpected_keys}")
    else:
        logger.warning(f"模型目录中没有找到.pth文件: {args.model_path}")
    
    # 全模型调参，不冻结任何参数
    trainable_count = sum(1 for param in model.parameters() if param.requires_grad)
    total_count = sum(1 for param in model.parameters())
    
    if args.local_rank in [-1, 0]:
        logger.info(f"总参数数量: {total_count}")
        logger.info(f"可训练参数数量: {trainable_count}")
        logger.info("训练策略: 全模型调参（包括ffn参数）")
    
    # 加载tokenizer
    vocab_file = os.path.join(args.model_path, "rwkv_vocab_enlarged.txt")
    logger.info(f"使用模型目录中的词汇表文件: {vocab_file}")
    tokenizer = RWKV_TOKENIZER(vocab_file)
    
    # 加载数据集
    train_dataset = load_and_prepare_dataset(model_args)
    
    if args.local_rank in [-1, 0]:
        logger.info(f"数据集大小: {len(train_dataset)}")
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=model_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=model_args.dataloader_pin_memory,
        num_workers=8,
    )
    
    # 配置优化器
    optimizer = configure_optimizer(model, model_args)
    
    # 初始化DeepSpeed引擎
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=model_args,
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )
    
    # 设置训练模式
    model_engine.train()
    
    # 初始化wandb
    if args.local_rank in [-1, 0]:
        wandb.init(project="rwkv7s2s-single-ffn-asr", name="asr-training-simplified")
    
    # 训练循环
    global_step = 0
    total_steps = len(train_dataloader) * model_args.num_train_epochs
    all_tokens = 0
    
    for epoch in range(model_args.num_train_epochs):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{model_args.num_train_epochs}")
        sum_loss = 0
        for step, batch in enumerate(progress_bar):
            # 创建输入和标签
            if step == 0 and is_main_process:
                print(f"batch: {batch}")
            
            input_ids, labels, attention_mask = create_asr_inputs_and_labels_simple(
                batch, tokenizer, model_args.eos_token_id, model_args.pad_token_id
            )
            
            if step == 0 and is_main_process:
                print(f"input_ids: {input_ids}")
                print(f"labels: {labels}")
                print(f"attention_mask: {attention_mask}")
            
            B, T = input_ids.shape
            current_tokens = B * T
            
            # 限制批次大小
            max_batch = (args.max_k_tokens_per_batch * 1024) // T
            if B > max_batch:
                print(f"当前批次序列长度 {B}，总tokens {current_tokens} 超过最大限制，截断!")
                input_ids = input_ids[:max_batch, :]
                labels = labels[:max_batch, :]
                attention_mask = attention_mask[:max_batch, :]
            
            all_tokens += input_ids.shape[0] * input_ids.shape[1]
            
            # 移动到设备
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            
            # 前向传播和反向传播
            loss = train_step(
                model_engine, 
                input_ids, 
                labels, 
                attention_mask
            )
            
            sum_loss += loss.item()
            avg_loss = sum_loss / (step + 1)
            
            # 反向传播
            model_engine.backward(loss)
            model_engine.step()
            
            # 添加调试信息
            if step % 100 == 0 and is_main_process:
                print(f"Step {step}: Loss: {loss.item():.4f}")
                print(f"  Batch size: {B}, Tokens processed: {input_ids.shape[0] * input_ids.shape[1]}")
            
            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "avg_loss": f"{avg_loss:.4f}"
            })
            
            # 更新学习率
            update_learning_rate(
                optimizer, 
                global_step, 
                total_steps, 
                model_args.warmup_steps, 
                model_args.learning_rate, 
                model_args.learning_rate_final, 
                model_args, 
                is_main_process
            )
            
            # 记录指标
            if global_step % model_args.logging_steps == 0:
                current_lr = optimizer.param_groups[0]['lr']
                log_metrics(global_step, loss.item(), current_lr, args, avg_loss, all_tokens)
            
            # 保存检查点
            if global_step % model_args.save_steps == 0 and global_step > 0:
                save_checkpoint(model_engine, epoch, global_step, model_args, loss.item(), avg_loss, is_final_step=False)
            
            global_step += 1
    
    # 保存最终模型
    if args.local_rank in [-1, 0]:
        save_checkpoint(model_engine, model_args.num_train_epochs, global_step, model_args, loss.item(), avg_loss, is_final_step=True)
        logger.info("训练完成！")

if __name__ == "__main__":
    main()
