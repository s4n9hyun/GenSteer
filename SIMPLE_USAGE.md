# Simple CALM Training

The train.sh script is now simplified for easy use.

## Basic Usage

```bash
./train.sh [MODEL] [DATASET] [BOTTLENECK_DIM]
```

## Examples

### 1. Default (LLaMA-7B + HH-RLHF)
```bash
./train.sh
```

### 2. Qwen3-8B with UltraFeedback
```bash
./train.sh "Qwen/Qwen3-8B" "HuggingFaceH4/ultrafeedback_binarized" 64
```

### 3. Qwen3-32B with UltraFeedback
```bash
./train.sh "Qwen/Qwen3-32B" "HuggingFaceH4/ultrafeedback_binarized" 128
```

### 4. LLaMA with different bottleneck
```bash
./train.sh "argsearch/llama-7b-sft-float32" "Dahoas/full-hh-rlhf" 64
```

## What it does

- Uses sensible defaults for all hyperparameters
- Automatically generates experiment names
- Creates output directories
- Runs training with accelerate
- Reports success/failure

## Default Settings

- **Epochs**: 1
- **Batch Size**: 16 per device  
- **Gradient Accumulation**: 2 (effective batch = 32)
- **Learning Rate**: 1e-5
- **Beta**: 0.1 (DPO parameter)
- **Regularization**: L2=0.001, Strength Variance=0.01, Entropy=0.01

## Outputs

Models are saved to: `./outputs/calm/calm-MODEL-DATASET-dimX/`

That's it! Simple and clean.