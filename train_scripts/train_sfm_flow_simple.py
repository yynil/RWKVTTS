import os
import torch
from torch.utils.data import DataLoader
import wandb
from dataclasses import dataclass, field
import logging
import json
from typing import Optional
import time
import numpy as np
import torchaudio
import onnxruntime
import whisper
import torchaudio.compliance.kaldi as kaldi
from torch.nn.utils.rnn import pad_sequence

from data.spark.multiple_webdataset import MultipleWebDataset
from hyperpyyaml import load_hyperpyyaml

logger = logging.getLogger(__name__)

@dataclass
class ScriptArguments:
    """Command line arguments for training script"""
    webdataset_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to webdataset directory"}
    )
    config_file: str = field(
        default=None,
        metadata={"help": "Path to model config file (YAML)"}
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
        default=4,
        metadata={"help": "Training batch size per device"}
    )
    learning_rate: float = field(
        default=1e-4,
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
    logging_steps: int = field(
        default=10,
        metadata={"help": "Number of steps between logging"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Number of steps between saving checkpoints"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    wandb_project: str = field(
        default="sfm-flow-training",
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
    ckpt_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model checkpoint file"}
    )
    campplus_path: str = field(
        default='/home/yueyulin/models/CosyVoice2-0.5B/campplus.onnx',
        metadata={"help": "Path to CamPPlus model"}
    )
    speech_tokenizer_file: str = field(
        default='/home/yueyulin/models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx',
        metadata={"help": "Path to speech tokenizer model"}
    )

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if 'LOG_LEVEL' not in os.environ else os.environ['LOG_LEVEL'],
    )

def configure_optimizer(model, args):
    """Configure optimizer with different learning rates for different parameter groups"""
    lr_1x = set()
    lr_2x = set()
    lr_decay = set()
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'sfm_head' in n:  # SFM head parameters get higher learning rate
            lr_2x.add(n)
        elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0) and (".weight" in n):
            lr_decay.add(n)
        else:
            lr_1x.add(n)

    lr_1x = sorted(list(lr_1x))
    lr_2x = sorted(list(lr_2x))
    lr_decay = sorted(list(lr_decay))
    param_dict = {n: p for n, p in model.named_parameters()}
    
    optim_groups = [
        {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "lr": args.learning_rate},
        {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "lr": args.learning_rate * 2.0}
    ]
    if args.weight_decay > 0:
        optim_groups.append({
            "params": [param_dict[n] for n in lr_decay],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate
        })
    
    optimizer = torch.optim.AdamW(optim_groups, betas=(0.9, 0.95), eps=1e-18)
    return optimizer

def update_learning_rate(optimizer, current_step, total_steps, warmup_steps, learning_rate, learning_rate_final, args):
    """Update learning rate for each parameter group"""
    import math
    
    # Calculate base learning rate
    if current_step < warmup_steps:
        base_lr = learning_rate * (0.01 + 0.99 * current_step / warmup_steps)
    else:
        # 使用current_step而不是total_steps来计算进度
        estimated_total_steps = args.num_epochs * 1000  # 可以根据实际情况调整
        progress = float(current_step - warmup_steps) / float(max(1, estimated_total_steps - warmup_steps))
        progress = max(0, min(1, progress))
        lr_final_factor = learning_rate_final / learning_rate
        base_lr = learning_rate * ((0.5 + lr_final_factor / 2) + (0.5 - lr_final_factor / 2) * math.cos(math.pi * progress))
    
    # Update learning rate for each parameter group
    for param_group in optimizer.param_groups:
        if 'sfm_head' in str(param_group['params'][0]):
            param_group['lr'] = base_lr * 2.0  # SFM head uses 2x learning rate
        else:
            param_group['lr'] = base_lr
        if current_step % 100 == 0:
            print(f'param_group lr: {param_group["lr"]} weight_decay: {param_group.get("weight_decay", 0)}')

def log_metrics(loss_dict, avg_loss, epoch, total_steps, kts, all_tokens, current_lr):
    """Log training metrics to wandb"""
    # Log basic training metrics
    wandb.log({
        "loss": loss_dict['loss'].item(),
        "avg_loss": avg_loss,
        "loss_coarse": loss_dict['loss_coarse'].item(),
        "loss_t": loss_dict['loss_t'].item(),
        "loss_sigma": loss_dict['loss_sigma'].item(),
        "loss_cfm_mu": loss_dict['loss_cfm_mu'].item(),
        "epoch": epoch,
        "step": total_steps,
        "KT/s": kts,
        "Gtokens": all_tokens/1e9,
        "learning_rate": current_lr
    })

def train_step(model, batch, device):
    """Execute one training step"""
    # Forward pass
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):  
        loss_dict = model(batch=batch, device=device)
    return loss_dict

def simple_collate_fn(batch):
    """Simple collate function that returns raw data list"""
    return batch

def process_one_batch(batch, feat_extractor, campplus_session, speech_tokenizer_session, sample_rate, device):
    """Process one batch of data"""
    mel_list = []
    embedding_list = []
    speech_token_list = []
    texts_list = []
    
    for i, data in enumerate(batch):
        speech = data['audio']['array']
        speech = torch.from_numpy(speech).unsqueeze(0).to(torch.float).to(device)
        mel = feat_extractor(speech).squeeze(0)
        
        if sample_rate != 16000:
            speech_16k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(speech.cpu())
        else:
            speech_16k = speech
        
        # Process CamPPlus embedding
        feat = kaldi.fbank(speech_16k, num_mel_bins=80, dither=0, sample_frequency=160000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        feat = feat.unsqueeze(0)
        embedding = campplus_session.run(None, {campplus_session.get_inputs()[0].name: feat.numpy()})[0].flatten().tolist()
        embedding_list.append(torch.tensor(embedding, dtype=torch.bfloat16))

        # Process speech tokenizer
        feat_whisper = whisper.log_mel_spectrogram(speech_16k, n_mels=128)
        feat_whisper_np = feat_whisper.detach().cpu().numpy()
        feat_length = np.array([feat_whisper.shape[2]], dtype=np.int32)
        
        speech_token = speech_tokenizer_session.run(None, {
            speech_tokenizer_session.get_inputs()[0].name: feat_whisper_np,
            speech_tokenizer_session.get_inputs()[1].name: feat_length
        })[0]
        speech_token = speech_token.squeeze(0)
        
        mel = mel.transpose(0, 1)
        token_mel_ratio = 2
        # Trim to align speech_token and speech_feat
        token_len = int(min(mel.shape[0] / token_mel_ratio, speech_token.shape[0]))
        speech_token = speech_token[:token_len]
        mel = mel[:token_len*token_mel_ratio]
        
        mel_list.append(mel)
        speech_token_list.append(speech_token)
        texts_list.append(data['json']['text'])
    
    return mel_list, embedding_list, speech_token_list, texts_list

def create_batch_data(mel_list, embedding_list, speech_token_list, texts_list, device):
    """Create batch data format"""
    batch_size = len(mel_list)
    
    # Process speech_feat (mel_list) - need padding
    feat_lengths = [mel.shape[1] for mel in mel_list]
    mel_sequences = [mel for mel in mel_list]
    speech_feat = pad_sequence(mel_sequences, batch_first=True, padding_value=0).to(device)
    speech_feat_len = torch.tensor(feat_lengths, dtype=torch.long, device=device)
    
    # Process speech_token - need padding
    token_lengths = [len(tokens) for tokens in speech_token_list]
    token_sequences = [torch.tensor(tokens, dtype=torch.long) for tokens in speech_token_list]
    speech_token = pad_sequence(token_sequences, batch_first=True, padding_value=0).to(device)
    speech_token_len = torch.tensor(token_lengths, dtype=torch.long, device=device)
    
    # Process embedding - stack directly
    embedding = torch.stack([emb.detach().clone() for emb in embedding_list], dim=0).to(device)
    
    # Validate embedding dimension
    expected_embedding_dim = 192
    if embedding.shape[1] != expected_embedding_dim:
        raise ValueError(f"Embedding dimension mismatch! Expected {expected_embedding_dim}, got {embedding.shape[1]}")
    
    return {
        'speech_feat': speech_feat.to(torch.bfloat16),
        'speech_feat_len': speech_feat_len,
        'speech_token': speech_token,
        'speech_token_len': speech_token_len,
        'embedding': embedding,
        'texts': texts_list
    }

def save_checkpoint(model, optimizer, output_dir, epoch, step, logger):
    """Save model checkpoint"""
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save model state dict
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    # Save optimizer state dict
    torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))

def main():
    # Parse arguments
    from transformers import HfArgumentParser
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args)
    )
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Arguments: {args}")

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load model config
    logger.info(f"Loading config from {args.config_file}")
    
    with open(args.config_file, 'r') as f:
        configs = load_hyperpyyaml(f)
    
    sample_rate = configs['sample_rate']
    feat_extractor = configs['feat_extractor']
    
    # Initialize ONNX sessions
    logger.info(f"Initializing ONNX sessions")
    
    # CamPPlus session
    providers = [("CUDAExecutionProvider", {"device_id": 0})]
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    campplus_session = onnxruntime.InferenceSession(args.campplus_path, sess_options=option, providers=providers)
    
    # Speech tokenizer session
    speech_tokenizer_session = onnxruntime.InferenceSession(args.speech_tokenizer_file, sess_options=option, providers=providers)
    
    # Load dataset
    if args.webdataset_dir is not None:
        logger.info(f"Loading dataset from webdataset directory: {args.webdataset_dir}")
        dataset = MultipleWebDataset(
            data_dir=args.webdataset_dir,
            target_sr=sample_rate,
            target_channels=1,
            shuffle=False,
            verify_tar=False
        )
    else:
        raise ValueError("webdataset_dir must be set")
    
    # Initialize model
    logger.info(f"Initializing model")
    
    flow_model = configs['flow']
    flow_model = flow_model.to(torch.bfloat16).to(device)
    
    # Convert all model parameters to bfloat16
    def convert_to_bfloat16(module):
        for param in module.parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)
        for buffer in module.buffers():
            if buffer.dtype == torch.float32:
                buffer.data = buffer.data.to(torch.bfloat16)
    
    # Apply bfloat16 conversion to all submodules
    flow_model.apply(convert_to_bfloat16)
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        flow_model.use_checkpoint = True
        logger.info("Gradient checkpointing enabled for SFM model")
    
    flow_model.train()
    
    if args.ckpt_file is not None:
        logger.info(f"Loading checkpoint from {args.ckpt_file}")
        info = flow_model.load_state_dict(torch.load(args.ckpt_file, map_location=device))
        logger.info(f"Loaded checkpoint info: {info}")
    
    flow_model.train()
    
    logger.info(f'Enable gradient checkpointing: {args.gradient_checkpointing}')
    
    for n, p in flow_model.named_parameters():
        p.requires_grad = True
    
    for n, p in flow_model.named_parameters():
        print(f'{n} requires grad: {p.requires_grad}')
    logger.info(f'start configuring optimizer')
    
    optimizer = configure_optimizer(flow_model, args)
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=simple_collate_fn,
        persistent_workers=True
    )
    
    # Delete output_dir if it exists
    if os.path.exists(args.output_dir):
        import shutil
        shutil.rmtree(args.output_dir)
    
    total_loss = 0.0
    total_steps = 0
    all_tokens = 0
    global_steps = 0
    
    for epoch in range(args.num_epochs):
        flow_model.train()
        update_time = time.time()
        logger.info(f"Epoch {epoch} starts training")
        from tqdm import tqdm
        pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(dataloader):
            # Update learning rate
            update_learning_rate(
                optimizer,
                global_steps,
                total_steps,
                args.warmup_steps,
                args.learning_rate,
                args.learning_rate_final,
                args
            )
            
            # Process batch data
            mel_list, embedding_list, speech_token_list, texts_list = process_one_batch(
                batch, feat_extractor, campplus_session, speech_tokenizer_session, sample_rate, device
            )
            
            batch_data = create_batch_data(mel_list, embedding_list, speech_token_list, texts_list, device)
            
            # Training step
            optimizer.zero_grad()
            loss_dict = train_step(flow_model, batch_data, device)
            loss = loss_dict['loss']
            
            if batch_idx == 0:
                print(f'loss: {loss}')
                print(f'loss components: {loss_dict}')
            
            global_steps += 1
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN loss detected at batch {batch_idx}, skipping this batch")
                wandb.log({
                    "nan_detected": 1,
                    "epoch": epoch,
                    "step": total_steps
                })
                continue
            else:
                # Normal case, use actual loss
                loss.backward()
                optimizer.step()
            
            if batch_idx % args.save_steps == 0 and batch_idx > 0:
                save_checkpoint(flow_model, optimizer, args.output_dir, epoch, batch_idx, logger)
            
            # Accumulate statistics
            elapsed_time = time.time() - update_time
            total_loss += loss.item()
            total_steps += 1
            
            # Calculate averages
            avg_loss = total_loss / total_steps
            tokens = batch_data['speech_token'].shape[1]
            all_tokens += tokens
            kts = tokens / elapsed_time / 1e3
            
            # Log to wandb
            current_lr = optimizer.param_groups[0]['lr']
            log_metrics(loss_dict, avg_loss, epoch, total_steps, kts, all_tokens, current_lr)
            
            pbar.update(1)
            pbar.set_postfix({
                'loss': loss.item(),
                'avg_loss': avg_loss,
                'lr': current_lr
            })
        
        # Save checkpoint at the end of each epoch
        epoch_checkpoint_dir = f"{args.output_dir}/epoch_{epoch}"
        os.makedirs(epoch_checkpoint_dir, exist_ok=True)
        print(f'saving checkpoint to {epoch_checkpoint_dir}')
        torch.save(flow_model.state_dict(), os.path.join(epoch_checkpoint_dir, 'model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(epoch_checkpoint_dir, 'optimizer.pt'))
        
        pbar.close()
    
    # Close wandb after training
    wandb.finish()

if __name__ == "__main__":
    main() 