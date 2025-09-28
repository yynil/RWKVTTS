# SFM Flow Training Guide

This guide explains how to train the Shallow Flow Matching (SFM) model using the provided configuration files and training scripts.

## Overview

The SFM model consists of several key components:
1. **CausalMaskedDiffWithXvecSFM** - Main SFM model
2. **UpsampleConformerEncoder** - Text encoder that processes speech tokens
3. **SFMHead** - Shallow Flow Matching head that predicts intermediate states
4. **CausalConditionalCFM** - Flow matching decoder with SFM support
5. **CausalConditionalDecoder** - UNet-style estimator for flow matching

## Training Scripts

### 1. `train_sfm_flow.py` - DeepSpeed Version
- Full-featured training script with DeepSpeed support
- Recommended for production training with multiple GPUs
- Supports distributed training and advanced memory optimization
- **Gradient Checkpointing**: Enabled via `--gradient_checkpointing True`

### 2. `train_sfm_flow_simple.py` - Simple Version
- Simplified training script without DeepSpeed dependency
- Recommended for single GPU training and quick experiments
- Easier to debug and modify
- **Gradient Checkpointing**: Enabled via `--gradient_checkpointing True`

## Configuration Files

### 1. `train_sfm_flow.yaml` - Standard Configuration
- Full-featured configuration for production training
- Recommended for final model training

### 2. `train_sfm_flow_detailed.yaml` - Detailed Configuration
- Comprehensive configuration with detailed comments
- Includes training optimization suggestions

### 3. `train_sfm_flow_simple.yaml` - Simplified Configuration
- Simplified model structure for faster training
- Recommended for debugging and initial experiments

## Usage Examples

### Using Simple Training Script (Recommended for Single GPU)

```bash
# Quick experiment with simple training script
python train_scripts/train_sfm_flow_simple.py \
    --config_file model/flow/train_sfm_flow_simple.yaml \
    --webdataset_dir /path/to/dataset \
    --output_dir /path/to/save/model \
    --num_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-4 \
    --gradient_checkpointing True \
    --wandb_project "sfm-experiment" \
    --wandb_run_name "sfm-simple-test"
```

### Using DeepSpeed Training Script (Recommended for Multi-GPU)

```bash
# Multi-GPU training with DeepSpeed
deepspeed --num_gpus=2 train_scripts/train_sfm_flow.py \
    --config_file model/flow/train_sfm_flow.yaml \
    --webdataset_dir /path/to/dataset \
    --output_dir /path/to/save/model \
    --num_epochs 10 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --gradient_checkpointing True \
    --ds_stage 3 \
    --ds_param_offload True \
    --ds_optimizer_offload True
```

## Key Differences Between Training Scripts

| Feature | Simple Version | DeepSpeed Version |
|---------|----------------|-------------------|
| **Dependencies** | Standard PyTorch | DeepSpeed + PyTorch |
| **GPU Support** | Single GPU | Multi-GPU |
| **Memory Optimization** | Basic | Advanced (ZeRO, offload) |
| **Distributed Training** | No | Yes |
| **Gradient Checkpointing** | Yes | Yes |
| **Mixed Precision** | Manual (bfloat16) | Automatic |
| **Checkpoint Format** | PyTorch .pt | DeepSpeed format |
| **Ease of Debugging** | High | Medium |

## Training Parameters

### Essential Parameters
- `--config_file`: Path to YAML configuration file
- `--webdataset_dir`: Path to WebDataset directory
- `--output_dir`: Directory to save checkpoints
- `--num_epochs`: Number of training epochs
- `--per_device_train_batch_size`: Batch size per GPU

### Learning Rate Parameters
- `--learning_rate`: Initial learning rate (default: 1e-4)
- `--learning_rate_final`: Final learning rate (default: 1e-6)
- `--warmup_steps`: Number of warmup steps (default: 100)

### Memory Optimization
- `--gradient_checkpointing`: Enable gradient checkpointing
- `--ds_stage`: DeepSpeed ZeRO stage (1, 2, or 3)
- `--ds_param_offload`: Offload parameters to CPU
- `--ds_optimizer_offload`: Offload optimizer states to CPU

### Monitoring
- `--wandb_project`: Weights & Biases project name
- `--wandb_run_name`: Weights & Biases run name
- `--logging_steps`: Steps between logging
- `--save_steps`: Steps between checkpoint saves

## Model Architecture Details

### SFM Head (2x Learning Rate)
The SFM head uses 2x learning rate because:
1. **Critical Component**: It's the core innovation that predicts intermediate states
2. **Gradient Flow**: Compensates for gradient decay through multiple layers
3. **Loss Characteristics**: Complex mathematical operations require higher learning rate
4. **Training Stability**: Helps maintain stable training dynamics

### Loss Components
- **L_coarse**: Coarse loss for initial predictions
- **L_t**: Time parameter prediction loss
- **L_sigma**: Variance parameter prediction loss  
- **L_cfm**: Conditional flow matching loss
- **L_mu**: Mean prediction loss

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're in the project root directory
   cd /path/to/RWKVTTS
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

2. **Memory Issues**
   ```bash
   # Reduce batch size
   --per_device_train_batch_size 1
   
   # Enable gradient checkpointing
   --gradient_checkpointing True
   
   # Use DeepSpeed with parameter offload
   --ds_stage 3 --ds_param_offload True
   ```

3. **NaN Loss**
   - The training script automatically detects and skips NaN losses
   - Check learning rate and model initialization
   - Verify input data preprocessing

4. **Slow Training**
   - Use DeepSpeed for multi-GPU training
   - Enable gradient checkpointing for memory efficiency
   - Adjust batch size based on available memory

## Performance Tips

1. **For Single GPU**: Use `train_sfm_flow_simple.py` with gradient checkpointing
2. **For Multi-GPU**: Use `train_sfm_flow.py` with DeepSpeed ZeRO stage 3
3. **Memory Optimization**: Enable parameter and optimizer offload
4. **Monitoring**: Use Weights & Biases for training visualization
5. **Checkpointing**: Save checkpoints regularly for recovery

## Expected Training Time

- **Single GPU (RTX 4090)**: ~2-3 hours per epoch (depending on dataset size)
- **Multi-GPU (2x A100)**: ~1-1.5 hours per epoch
- **Full Training**: 10-20 epochs for convergence

## Model Checkpoints

### Simple Version Format
```
output_dir/
├── epoch_0_step_500/
│   ├── model.pt
│   └── optimizer.pt
├── epoch_1/
│   ├── model.pt
│   └── optimizer.pt
```

### DeepSpeed Version Format
```
output_dir/
├── epoch_0_step_500/
│   ├── mp_rank_00_model_states.pt
│   └── zero_pp_rank_0_mp_rank_00_optim_states.pt
├── epoch_1/
│   └── ...
``` 