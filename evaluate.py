#!/usr/bin/env python3
"""
Evaluate trained CALM model using the same evaluation methodology as training.
"""

import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator
from tqdm import tqdm

from models import create_calm_model
from data import PreferenceDataset, load_preference_data
from train import AlignmentLoss

def evaluate_calm_model(
    checkpoint_path,
    model_name="Qwen/Qwen3-8B",
    dataset_name="HuggingFaceH4/ultrafeedback_binarized",
    dataset_split="test_prefs",
    bottleneck_dim=32,
    batch_size=4,
    max_length=1024
):
    """Evaluate CALM model on test dataset."""
    
    print("🔍 CALM Model Evaluation")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name} ({dataset_split})")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision="bf16")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load evaluation dataset
    print(f"Loading evaluation dataset: {dataset_name} ({dataset_split})")
    eval_data = load_preference_data(dataset_name, split=dataset_split)
    print(f"Loaded {len(eval_data)} evaluation samples")
    
    # Create evaluation dataset
    eval_dataset = PreferenceDataset(eval_data, tokenizer, max_length)
    
    # Define collate function
    def collate_fn(batch):
        return {
            "chosen_input_ids": torch.stack([item["input_ids_chosen"] for item in batch]),
            "chosen_attention_mask": torch.stack([item["attention_mask_chosen"] for item in batch]),
            "rejected_input_ids": torch.stack([item["input_ids_rejected"] for item in batch]),
            "rejected_attention_mask": torch.stack([item["attention_mask_rejected"] for item in batch]),
        }
    
    # Create data loader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    # Create CALM model
    print(f"Creating CALM model with bottleneck dim {bottleneck_dim}...")
    model = create_calm_model(
        base_model_name=model_name,
        bottleneck_dim=bottleneck_dim,
        max_modulation_strength=5.0,
        device=accelerator.device,
        torch_dtype=torch.bfloat16
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=accelerator.device, weights_only=False)
    
    if "alignment_model" in checkpoint:
        model.modulation_engine.load_state_dict(checkpoint["alignment_model"])
    else:
        raise KeyError(f"Cannot find alignment_model in checkpoint keys: {checkpoint.keys()}")
    
    print("✅ Checkpoint loaded successfully")
    
    # Setup loss function
    loss_fn = AlignmentLoss(
        beta=0.1,
        lambda_l2=0.001,
        lambda_strength_variance=0.01,
        lambda_entropy=0.01
    )
    
    # Prepare with accelerator
    model, eval_loader = accelerator.prepare(model, eval_loader)
    
    # Run evaluation
    model.eval()
    eval_losses = []
    all_modulation_strengths = []
    
    print("🔍 Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            # Forward pass for chosen responses
            chosen_outputs = model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
                return_components=True
            )
            
            # Forward pass for rejected responses
            rejected_outputs = model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
                return_components=True
            )
            
            # Compute loss
            loss, loss_components = loss_fn(chosen_outputs, rejected_outputs, batch)
            eval_losses.append(loss_components["total_loss"])
            
            # Collect modulation statistics
            if "modulation_strength" in chosen_outputs:
                chosen_strengths = chosen_outputs["modulation_strength"].detach().cpu().numpy()
                rejected_strengths = rejected_outputs["modulation_strength"].detach().cpu().numpy()
                all_modulation_strengths.extend(chosen_strengths.flatten())
                all_modulation_strengths.extend(rejected_strengths.flatten())
    
    # Calculate evaluation metrics
    avg_eval_loss = np.mean(eval_losses)
    
    print(f"\n📊 Evaluation Results:")
    print(f"  Average Loss: {avg_eval_loss:.4f}")
    print(f"  Total Samples: {len(eval_data)}")
    print(f"  Batches Processed: {len(eval_losses)}")
    
    if all_modulation_strengths:
        print(f"\n📈 Modulation Statistics:")
        print(f"  Mean Strength: {np.mean(all_modulation_strengths):.3f}")
        print(f"  Std Strength:  {np.std(all_modulation_strengths):.3f}")
        print(f"  Min Strength:  {np.min(all_modulation_strengths):.3f}")
        print(f"  Max Strength:  {np.max(all_modulation_strengths):.3f}")
        print(f"  Active (>0.5): {np.mean(np.array(all_modulation_strengths) > 0.5):.1%}")
        print(f"  High (>2.0):   {np.mean(np.array(all_modulation_strengths) > 2.0):.1%}")
    
    return {
        "avg_loss": avg_eval_loss,
        "modulation_stats": {
            "mean": np.mean(all_modulation_strengths) if all_modulation_strengths else 0,
            "std": np.std(all_modulation_strengths) if all_modulation_strengths else 0,
            "min": np.min(all_modulation_strengths) if all_modulation_strengths else 0,
            "max": np.max(all_modulation_strengths) if all_modulation_strengths else 0,
        } if all_modulation_strengths else {}
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate CALM model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to CALM checkpoint")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B",
                       help="Base model name")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrafeedback_binarized",
                       help="Dataset name")
    parser.add_argument("--dataset_split", type=str, default="test_prefs",
                       help="Dataset split")
    parser.add_argument("--bottleneck_dim", type=int, default=32,
                       help="Bottleneck dimension")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Max sequence length")
    
    args = parser.parse_args()
    
    evaluate_calm_model(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        bottleneck_dim=args.bottleneck_dim,
        batch_size=args.batch_size,
        max_length=args.max_length
    )