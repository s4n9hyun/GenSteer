#!/bin/bash

# CALM Training Script
# Paper: CALM: Controllable Alignment via Logit Modulation

# Default configuration
MODEL_NAME="argsearch/llama-7b-sft-float32"
DATASET_NAME="Dahoas/full-hh-rlhf"
BOTTLENECK_DIM=32

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --bottleneck_dim)
            BOTTLENECK_DIM="$2"
            shift 2
            ;;
        *)
            # Handle positional arguments for backward compatibility
            if [[ -z "${MODEL_NAME_SET:-}" ]]; then
                MODEL_NAME="$1"
                MODEL_NAME_SET=1
            elif [[ -z "${DATASET_NAME_SET:-}" ]]; then
                DATASET_NAME="$1"
                DATASET_NAME_SET=1
            elif [[ -z "${BOTTLENECK_DIM_SET:-}" ]]; then
                BOTTLENECK_DIM="$1"
                BOTTLENECK_DIM_SET=1
            fi
            shift
            ;;
    esac
done

# Basic settings
EPOCH=1
BATCH_SIZE=4   # Reduced further for multi-GPU setup
GRAD_ACCUM=4   # Increased to maintain effective batch size
LR=1e-5
BETA=0.1

# Generate experiment name
MODEL_SHORT=$(echo $MODEL_NAME | cut -d'/' -f2)
DATASET_SHORT=$(echo $DATASET_NAME | cut -d'/' -f2)
EXP_NAME="calm-${MODEL_SHORT}-${DATASET_SHORT}-dim${BOTTLENECK_DIM}"

OUTPUT_DIR="./outputs/calm/${EXP_NAME}"

echo "🎯 Training CALM - Controllable Alignment via Logit Modulation"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "GPUs: 2 (multi-GPU training enabled)"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training with multi-GPU support
accelerate launch --num_processes=2 --multi_gpu train.py \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs $EPOCH \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --learning_rate $LR \
    --beta $BETA \
    --bottleneck_dim $BOTTLENECK_DIM \
    --lambda_l2 0.001 \
    --lambda_strength_variance 0.01 \
    --lambda_entropy 0.01 \
    --warmup_steps 200 \
    --save_steps 1000 \
    --eval_steps 100000 \
    --max_length 1024 \
    --bf16 \
    --seed 42

if [ $? -eq 0 ]; then
    echo "✅ Training completed!"
    echo "📁 Model saved to: $OUTPUT_DIR"
else
    echo "❌ Training failed"
fi