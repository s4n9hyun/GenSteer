# CALM with Qwen Models and UltraFeedback Dataset

This guide shows how to train CALM with Qwen models using the UltraFeedback dataset.

## Supported Models

- **Qwen/Qwen3-8B**: 8B parameter multilingual model with strong reasoning capabilities
- **Qwen/Qwen3-32B**: 32B parameter model with advanced reasoning and agent capabilities

## Supported Datasets

- **HuggingFaceH4/ultrafeedback_binarized**: High-quality preference dataset with 187k samples
  - Splits: `train_prefs`, `test_prefs`, `train_sft`, `test_sft`

## Quick Start Examples

### 1. Train CALM with Qwen3-8B on UltraFeedback

```bash
./train.sh \
  --model "Qwen/Qwen3-8B" \
  --dataset "HuggingFaceH4/ultrafeedback_binarized" \
  --dataset_split "train_prefs" \
  --epoch 1 \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 5e-6 \
  --bottleneck_dim 64
```

### 2. Train CALM with Qwen3-32B on UltraFeedback

```bash
./train.sh \
  --model "Qwen/Qwen3-32B" \
  --dataset "HuggingFaceH4/ultrafeedback_binarized" \
  --dataset_split "train_prefs" \
  --epoch 1 \
  --batch_size 2 \
  --grad_accum 8 \
  --lr 2e-6 \
  --bottleneck_dim 128
```

### 3. Using Environment Variables

```bash
# For Qwen3-8B
export MODEL_NAME="Qwen/Qwen3-8B"
export DATASET_NAME="HuggingFaceH4/ultrafeedback_binarized"
export DATASET_SPLIT="train_prefs"
export BATCH_SIZE=4
export GRAD_ACCUM=4
export LR=5e-6
export BOTTLENECK_DIM=64

./train.sh
```

## Key Differences from LLaMA Training

### Model-Specific Adjustments

**For Qwen models:**
- Use lower learning rates (2e-6 to 5e-6 vs 1e-5 for LLaMA)
- May require larger bottleneck dimensions (64-128 vs 32)
- Support for multilingual training and evaluation

### Dataset-Specific Adjustments

**For UltraFeedback:**
- Use `train_prefs` split for preference training
- Higher quality responses than HH-RLHF
- Larger dataset size (187k vs 160k samples)

## Resource Requirements

### Qwen3-8B
- **VRAM**: ~16GB (with bf16)
- **Recommended**: Single A100/H100 or RTX 4090
- **Batch size**: 4-8 per device

### Qwen3-32B  
- **VRAM**: ~64GB (with bf16)
- **Recommended**: Multiple A100s or H100
- **Batch size**: 1-2 per device
- **Note**: May require model parallelism for single GPU setups

## Advanced Configuration

### Custom Hyperparameters

```bash
./train.sh \
  --model "Qwen/Qwen3-8B" \
  --dataset "HuggingFaceH4/ultrafeedback_binarized" \
  --dataset_split "train_prefs" \
  --beta 0.2 \
  --lambda_l2 0.002 \
  --lambda_strength_variance 0.02 \
  --lambda_entropy 0.02 \
  --bottleneck_dim 64 \
  --epoch 2
```

### Multi-GPU Training

```bash
# Automatically detects available GPUs
accelerate config  # Run this first
./train.sh --model "Qwen/Qwen3-32B" --batch_size 1
```

## Expected Results

### Training Performance
- **Qwen3-8B**: ~2-3 hours per epoch on A100
- **Qwen3-32B**: ~8-12 hours per epoch on A100

### Model Performance
Qwen models typically show:
- Strong multilingual capabilities
- Better reasoning on complex tasks
- Improved instruction following
- Enhanced agent-like behaviors

## Troubleshooting

### Memory Issues
```bash
# Reduce batch size
./train.sh --model "Qwen/Qwen3-32B" --batch_size 1 --grad_accum 16

# Use gradient checkpointing (add to train.py if needed)
# Enable CPU offloading for very large models
```

### Dataset Loading Issues
```bash
# Verify dataset access
python -c "from datasets import load_dataset; print(load_dataset('HuggingFaceH4/ultrafeedback_binarized', split='train_prefs'))"
```

### Tokenizer Issues
```bash
# Some Qwen models may need specific tokenizer handling
# Check tokenizer compatibility in train.py
```

## Performance Tips

1. **Use bf16**: Always enabled by default
2. **Gradient accumulation**: Increase if memory limited
3. **Learning rate**: Start with lower rates for Qwen models
4. **Bottleneck dimension**: Scale with model size (32→64→128)
5. **Evaluation frequency**: Set `--eval_steps` appropriately for dataset size

## Model Outputs

Trained models will be saved in:
```
./outputs/calm/calm-qwen3-8b-ultrafeedback-dim64-epoch_1-beta_0.1-lr_5e-6-l2_0.001-sv_0.01-ent_0.01/
```

With files:
- `best_calm.pt` - Best checkpoint
- `training_config.json` - Training configuration
- Model logs and metrics