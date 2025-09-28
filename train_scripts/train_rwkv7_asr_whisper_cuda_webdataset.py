#!/usr/bin/env python3
"""
基于RWKV7ASRModelCuda (Whisper版本) 的ASR训练脚本
使用webdataset数据，冻结Whisper组件，只训练LLM和projector
"""

import os
import sys
import json
import logging
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import deepspeed
import wandb
from tqdm import tqdm
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.llm.rwkv_asr_cuda_whisper import RWKV7ASRModelCuda, load_whisper_feature_extractor_and_encoder, RWKV7ModelForLatentInputsCuda, RWKV7ModelForCausalLMCuda
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
from utils.webdataset_utils import create_complete_pipeline, process_batch
from utils.rwkv_utilities import parser_config_from_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量，用于存储预编码的指令和hints
_global_text_input_ids_chinese = None
_global_text_input_ids_english = None
_global_hints_ids = None

def create_asr_inputs_and_labels(batch, tokenizer, eos_id=0, pad_id=0):
    """为ASR任务创建输入和标签，使用Whisper版本的RWKV7ASRModelCuda格式"""
    
    # 从webdataset_utils返回的batch格式：
    # batch = (wavs, texts, formats, languages, sample_rates)
    
    # 使用process_batch处理原始数据
    processed_batch = process_batch(batch, tokenizer, eos_token_id=0, pad_token_id=0)
    
    # 音频数据：直接使用numpy数组
    audio_data_list = list(processed_batch['wavs'])
    
    # 文本输入：使用预处理的instruction_tokens
    text_input_ids = processed_batch['instruction_tokens']
    text_attention_mask = processed_batch['text_attention_mask']
    
    # 标签：使用预处理的labels
    labels = processed_batch['labels']
    
    # 创建labels_attention_mask：所有非-100的位置都是有效的
    labels_attention_mask = (labels != -100).long()
    
    # hints_ids：使用预处理的hints_tokens（取第一个样本的hints）
    hints_ids = processed_batch['hints_tokens'][0] if len(processed_batch['hints_tokens']) > 0 else torch.tensor([], dtype=torch.long)
    
    return audio_data_list, text_input_ids, text_attention_mask, labels, labels_attention_mask, hints_ids

def train_step(model_engine, audio_data, text_input_ids, text_attention_mask, labels, labels_attention_mask, hints_ids):
    """训练步骤"""
    output = model_engine(audio_data, text_input_ids, text_attention_mask, labels, labels_attention_mask, hints_ids)
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
    """配置优化器，只训练LLM和projector，冻结Whisper组件"""
    lr_1x = set()
    lr_2x = set()
    lr_decay = set()    
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        # 冻结Whisper相关参数
        if ("whisper_encoder." in n or "whisper_feature_extractor." in n or "llm." in n) and not "audio_lm_model" in n:
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

def log_metrics(wandb, global_step, loss, avg_loss, learning_rate, all_tokens, epoch, is_main_process):
    """记录训练指标到wandb"""
    if is_main_process:
        wandb.log({
            "step": global_step,
            "epoch": epoch,
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

def load_model_from_checkpoint_dir(checkpoint_dir, device):
    """从checkpoint目录加载模型组件"""
    
    # 加载LLM参数
    llm_args_path = os.path.join(checkpoint_dir, 'llm_args.json')
    with open(llm_args_path, 'r') as f:
        llm_args_dict = json.load(f)
    
    # 创建Namespace对象
    from argparse import Namespace
    llm_args = Namespace(**llm_args_dict)
    
    # 加载LLM模型
    llm = RWKV7ModelForCausalLMCuda(llm_args)
    llm_ckpt_path = os.path.join(checkpoint_dir, 'llm_state_dict.pt')
    llm.load_state_dict(torch.load(llm_ckpt_path, map_location='cpu'))
    llm = llm.to(device)
    
    # 加载音频LM参数
    audio_lm_args_path = os.path.join(checkpoint_dir, 'audio_lm_args.json')
    with open(audio_lm_args_path, 'r') as f:
        audio_lm_args_dict = json.load(f)
    
    audio_lm_args = Namespace(**audio_lm_args_dict)
    
    # 加载音频LM模型
    audio_lm_model = RWKV7ModelForLatentInputsCuda(audio_lm_args)
    audio_lm_ckpt_path = os.path.join(checkpoint_dir, 'audio_lm_state_dict.pt')
    audio_lm_model.load_state_dict(torch.load(audio_lm_ckpt_path, map_location='cpu'))
    audio_lm_model = audio_lm_model.to(device)
    
    # 加载Whisper组件
    whisper_encoder_path = os.path.join(checkpoint_dir, 'whisper_encoder')
    whisper_feature_extractor, whisper_encoder = load_whisper_feature_extractor_and_encoder(whisper_encoder_path)
    whisper_encoder = whisper_encoder.to(device)
    
    return whisper_feature_extractor, whisper_encoder, audio_lm_model, llm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="/home/yueyulin/rwkv7_whisper_cuda", help="模型checkpoint目录路径")
    parser.add_argument("--data_dir", type=str, required=True, help="webdataset数据目录")
    parser.add_argument("--output_dir", type=str, default="outputs/rwkv7_asr_whisper_cuda", help="输出目录")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="每个设备的训练批次大小")
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
    parser.add_argument("--all_training_steps", type=int, default=1000000, help="总训练步数，WebDataset流式训练使用")
    parser.add_argument("--gradient_clipping", type=float, default=0.5, help="梯度裁剪阈值")
    
    args = parser.parse_args()
    
    # 初始化DeepSpeed
    deepspeed.init_distributed()
    
    # 获取分布式训练信息
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    
    # 设置设备
    device = torch.device("cuda", args.local_rank)
    global is_main_process
    is_main_process = args.local_rank in [-1, 0]
    
    if is_main_process:
        print(f"DeepSpeed初始化完成，world_size={world_size}, local_rank={args.local_rank}")
        print(f"使用设备: {device}")
    
    # 从checkpoint目录加载模型组件
    if is_main_process:
        print(f"从checkpoint目录加载模型: {args.checkpoint_dir}")
    
    whisper_feature_extractor, whisper_encoder, audio_lm_model, llm = load_model_from_checkpoint_dir(args.checkpoint_dir, device)
    
    # 创建ASR模型
    if is_main_process:
        print("创建ASR模型...")
    model = RWKV7ASRModelCuda(whisper_encoder, audio_lm_model, llm, whisper_feature_extractor)
    if args.gradient_checkpointing:
        model.audio_lm_model.args.grad_cp = 1
    model.whisper_encoder.eval()
    model.llm.eval()
    # 冻结Whisper相关参数
    for name, param in model.named_parameters():
        if ("whisper_encoder." in name or "whisper_feature_extractor." in name or "llm." in name) and not "audio_lm_model" in name:
            param.requires_grad = False
            if is_main_process:
                print(f"冻结参数: {name}")
    
    model.train()
    if is_main_process:
        for name, param in model.named_parameters():
            print(f"参数: {name} 形状: {param.shape} 是否可训练: {param.requires_grad}")
        print("ASR模型创建完成")
    
    # 从checkpoint恢复训练
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        if is_main_process:
            print(f"加载参数: {args.resume_from_checkpoint}")
        info = model.load_state_dict(torch.load(args.resume_from_checkpoint), strict=False)
        print(f"加载参数信息: {info}")
    
    if is_main_process:
        print(f"模型创建完成")
        print(f"总参数数量: {sum(1 for param in model.parameters())}")
        print(f"可训练参数数量: {sum(1 for param in model.parameters() if param.requires_grad)}")
        print(f"冻结参数数量: {sum(1 for param in model.parameters() if not param.requires_grad)}")
    
    # 从checkpoint恢复训练状态
    global_step = 0
    all_tokens = 0
    start_epoch = 0
    
    # 加载tokenizer
    vocab_file = os.path.join(args.checkpoint_dir, "rwkv_vocab_v20230424.txt")
    
    tokenizer = TRIE_TOKENIZER(vocab_file)
    
    # 初始化全局预编码变量
    if is_main_process:
        print("初始化全局预编码变量...")
    
    # 预定义的中英文指令
    chinese_instruction = "User: 请将以下语音转写为中文。\n"
    english_instruction = "User: Convert the audios to English.\n"
    # 预编码指令
    global _global_text_input_ids_chinese, _global_text_input_ids_english, _global_hints_ids
    _global_text_input_ids_chinese = torch.tensor(tokenizer.encode(chinese_instruction), dtype=torch.long)
    _global_text_input_ids_english = torch.tensor(tokenizer.encode(english_instruction), dtype=torch.long)
    _global_hints_ids = torch.tensor(tokenizer.encode("Assistant: "), dtype=torch.long)
    
    if is_main_process:
        print(f"中文指令长度: {len(_global_text_input_ids_chinese)}")
        print(f"英文指令长度: {len(_global_text_input_ids_english)}")
        print(f"hints长度: {len(_global_hints_ids)}")
    
    # 加载webdataset数据集
    if is_main_process:
        print(f"加载webdataset数据集: {args.data_dir}")
    
    # 查找tar文件
    import glob
    data_files = glob.glob(os.path.join(args.data_dir, "**/*.tar"))
    if not data_files:
        raise ValueError(f"No .tar files found in {args.data_dir}")
    
    if is_main_process:
        print(f"找到 {len(data_files)} 个tar文件")
        print(f"前5个文件: {data_files[:5]}")
    data_files = sorted(data_files)
    print(f"排序后的文件: {data_files}")
    # 创建webdataset数据集
    if is_main_process:
        print(f"创建WebDataset pipeline...")
    
    # 使用webdataset_utils创建完整的pipeline
    dataset, dataloader = create_complete_pipeline(
        data_files=data_files,
        world_size=world_size,
        global_rank=global_rank,  # 使用正确的全局rank
        batch_size=args.per_device_train_batch_size,
        target_sample_rate=16000,
        num_workers=4,
        shardshuffle=100
    )
    
    if is_main_process:
        print(f"WebDataset pipeline创建完成")
        print(f"数据加载器创建完成，使用WebDataset流式数据")
    
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
    
    # 初始化wandb
    if is_main_process:
        wandb.init(project="rwkv7-asr-whisper-cuda", name="whisper-asr-cuda-training")
    
    # 训练循环
    if is_main_process:
        print(f"开始训练循环，总训练步数: {args.all_training_steps}")

    # 创建数据加载器迭代器
    sum_loss = 0
    all_tokens = 0
    iter_dataloader = iter(dataloader)
    if is_main_process:
        progress_bar = tqdm(total=args.all_training_steps, desc="Training")
    # 初始化训练状态
    epoch = 0
    while global_step < args.all_training_steps:
        try:
            batch = next(iter_dataloader)
        except StopIteration:
            # 数据遍历完毕，重新创建迭代器开始新的epoch
            if is_main_process:
                print(f"Epoch {epoch} 完成，重新开始数据遍历...")
            epoch += 1
            iter_dataloader = iter(dataloader)
            batch = next(iter_dataloader)
            if is_main_process:
                print(f"开始 Epoch {epoch}")
        if global_step % 100 == 0:
            print(f"Rank {global_rank} Global Step {global_step}, batch: \n{batch}")

        wavs, texts, formats, languages, sample_rates = batch

        # 处理数据
        audio_data, text_input_ids, text_attention_mask, labels, labels_attention_mask, hints_ids = create_asr_inputs_and_labels(
                (wavs, texts, formats, languages, sample_rates), tokenizer, eos_id=0, pad_id=0
        )
        
        # 更新学习率
        update_learning_rate(
            optimizer,
            global_step,
            args.all_training_steps,
            args.warmup_steps,
            args.learning_rate,
            args.learning_rate_final,
            args,
            is_main_process
        )
        
        # 创建输入和标签
        try:
            # 估算token数量（音频特征 + 文本）
            # 音频特征：每个音频样本大约1500帧（Whisper下采样后）
            #calculate the audio actual length
            actual_length_of_audio = [0]
            index = 0
            for a in audio_data:
                actual_length_of_audio.append(actual_length_of_audio[index] + a.shape[1])
                index += 1
            actual_length_of_audio = actual_length_of_audio[-1] / 16000
            print(f"audio actual length: {actual_length_of_audio}")
            estimated_audio_tokens = int(actual_length_of_audio) * 50
            text_tokens = text_input_ids.shape[0] * text_input_ids.shape[1]
            labels_tokens = labels.shape[0] * labels.shape[1]
            current_k_tokens = (estimated_audio_tokens + text_tokens + labels_tokens) / 1024
            
            if current_k_tokens > args.max_k_tokens_per_batch:
                print(f"当前batch的token数超过最大限制: {current_k_tokens:.2f}k > {args.max_k_tokens_per_batch}k")
                # 调整batch大小
                max_bsz = int(args.max_k_tokens_per_batch * 1024 / (1500 + text_input_ids.shape[1]))
                max_bsz = max(1, max_bsz)  # 至少保留1个样本
                print(f"调整batch大小为: {max_bsz}")
                
                audio_data = audio_data[:max_bsz]
                text_input_ids = text_input_ids[:max_bsz, :]
                text_attention_mask = text_attention_mask[:max_bsz, :]
                labels = labels[:max_bsz, :]
                labels_attention_mask = labels_attention_mask[:max_bsz, :]
            
            if global_step == 0 and is_main_process:
                print(f"输入形状: audio_batch_size={len(audio_data)}, text={text_input_ids.shape}, labels={labels.shape}")
                
        except Exception as e:
            if is_main_process:
                print(f"数据处理失败: {e}")
                print(f"问题batch: {batch}")
            continue
        try:
            # 移动到设备
            text_input_ids = text_input_ids.to(device)
            text_attention_mask = text_attention_mask.to(device)
            labels = labels.to(device)
            labels_attention_mask = labels_attention_mask.to(device)
            hints_ids = hints_ids.to(device)
            
            # 前向传播和反向传播
            loss = train_step(model_engine, audio_data, text_input_ids, text_attention_mask, labels, labels_attention_mask, hints_ids)
            
            sum_loss += loss.item()
            avg_loss = sum_loss / (global_step + 1)
            
            # 反向传播
            model_engine.backward(loss)

            model_engine.step()
            
            # 计算总 tokens（包括音频和文本）
            text_tokens = text_input_ids.shape[0] * text_input_ids.shape[1] * world_size
            all_tokens += text_tokens
            
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录指标（每一步都记录）
            log_metrics(wandb, global_step, loss.item(), avg_loss, current_lr, all_tokens, epoch, is_main_process)
            
            # 更新进度条
            if is_main_process:
                progress_bar.set_postfix({
                    "epoch": epoch,
                    "loss": f"{loss.item():.4f}", 
                    "avg_loss": f"{avg_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "grad_clip": f"{args.gradient_clipping}",
                    "audio_actual_length": f"{actual_length_of_audio:.2f}"
                })
            
            # 保存checkpoint
            if global_step > 0 and global_step % args.save_steps == 0:
                training_state = get_training_state(model_engine, global_step, epoch, all_tokens)
                save_checkpoint(model_engine, args.output_dir, epoch, global_step, training_state, logger)
            
            global_step += 1
            
            # 更新进度条
            if is_main_process:
                progress_bar.update(1)
            
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"问题batch: {batch}")
            print(f"训练失败: {e}")
            print(f'use a dummy batch to recover...')
            torch.cuda.empty_cache()
            model_engine.step()
            continue
    
    # 关闭进度条
    if is_main_process:
        progress_bar.close()
    
    if is_main_process:
        logger.info("训练完成！")

if __name__ == "__main__":
    main()
