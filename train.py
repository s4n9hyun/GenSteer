#!/usr/bin/env python3
"""
GenSteer Training Script - Optimized for Efficiency
Paper: GenSteer: A Generative Steering Engine for Autonomous Test-Time Alignment

Key optimizations:
- Single forward pass for both chosen and rejected samples
- Efficient log probability calculation using cross-entropy
- Automatic mixed precision training
- Advanced regularization techniques
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# Import GenSteer models
from models import BaseLanguageModel, GenerativeSteeringEngine, GenSteer, create_gensteer


def prepare_preference_dataset(dataset, tokenizer, max_length=1024, num_proc=4):
    """Prepare preference dataset for GenSteer training."""
    
    def process_example(example):
        chosen = example.get("chosen", "")
        rejected = example.get("rejected", "")
        
        # Tokenize with padding
        chosen_tokens = tokenizer(
            chosen,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        rejected_tokens = tokenizer(
            rejected,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(),
        }
    
    # Process dataset
    processed = dataset.map(
        process_example,
        num_proc=num_proc,
        remove_columns=dataset.column_names
    )
    
    return processed


class GenSteerTrainer:
    """Optimized trainer for GenSteer models."""
    
    def __init__(
        self,
        model: GenSteer,
        tokenizer,
        args,
        accelerator: Accelerator
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.accelerator = accelerator
        
        # Loss weights
        self.beta = args.beta  # DPO temperature
        self.lambda_l2 = args.lambda_l2  # L2 regularization for steering vectors
        self.lambda_steering = args.lambda_steering  # Steering strength variance regularization
        self.lambda_entropy = args.lambda_entropy  # Entropy regularization
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
    def compute_dpo_loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor
    ) -> torch.Tensor:
        """Compute DPO loss with numerical stability."""
        
        # Compute log ratios
        chosen_rewards = self.beta * (chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (rejected_logps - reference_rejected_logps)
        
        # DPO loss with numerical stability
        rewards_diff = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(rewards_diff).mean()
        
        return loss, rewards_diff.detach().mean()
    
    def compute_steering_regularization(self, steering_strength: torch.Tensor) -> torch.Tensor:
        """Encourage variance in steering strength for better exploration."""
        steering_variance = torch.var(steering_strength)
        return -steering_variance  # Negative because we want to maximize variance
    
    def compute_entropy_regularization(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Prevent repetitive generation patterns."""
        
        # Get probability distribution
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Compute entropy: H = -sum(p * log(p))
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # Mask and average
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(entropy[..., None]).squeeze(-1)
            entropy = entropy * mask_expanded
            entropy = entropy.sum() / mask_expanded.sum()
        else:
            entropy = entropy.mean()
        
        # Return negative to maximize entropy (minimize negative entropy)
        return -entropy
    
    def compute_l2_regularization(self, steering_vector: torch.Tensor) -> torch.Tensor:
        """L2 regularization for steering vectors."""
        return torch.mean(torch.norm(steering_vector, p=2, dim=-1))
    
    def get_batch_log_probs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probabilities efficiently using cross-entropy."""
        
        batch_size, seq_len, vocab_size = logits.shape
        
        # Shift for autoregressive prediction
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        attention_mask = attention_mask[:, 1:].contiguous() if attention_mask is not None else None
        
        # Flatten for cross-entropy computation
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        # Efficient log probability calculation
        per_token_logps = -F.cross_entropy(
            logits_flat,
            labels_flat,
            reduction='none'
        ).view(batch_size, seq_len - 1)
        
        # Apply attention mask and sum
        if attention_mask is not None:
            per_token_logps = per_token_logps * attention_mask
            log_probs = per_token_logps.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
        else:
            log_probs = per_token_logps.mean(dim=1)
        
        return log_probs
    
    def training_step(self, batch: Dict) -> Dict:
        """Single optimized training step with combined forward pass."""
        
        # Extract batch data
        chosen_input_ids = batch["chosen_input_ids"]
        chosen_attention_mask = batch["chosen_attention_mask"]
        rejected_input_ids = batch["rejected_input_ids"]
        rejected_attention_mask = batch["rejected_attention_mask"]
        
        batch_size = chosen_input_ids.shape[0]
        
        # ========== Optimized Single Forward Pass ==========
        # Combine chosen and rejected for efficiency
        combined_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        combined_attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
        
        # Single forward pass through model
        combined_outputs = self.model(
            combined_input_ids,
            combined_attention_mask,
            return_components=True
        )
        
        # Split outputs
        chosen_outputs = {k: v[:batch_size] for k, v in combined_outputs.items()}
        rejected_outputs = {k: v[batch_size:] for k, v in combined_outputs.items()}
        
        # ========== Compute Log Probabilities ==========
        # GenSteer outputs (with steering)
        chosen_logps = self.get_batch_log_probs(
            chosen_outputs["logits"],
            chosen_input_ids,
            chosen_attention_mask
        )
        
        rejected_logps = self.get_batch_log_probs(
            rejected_outputs["logits"],
            rejected_input_ids,
            rejected_attention_mask
        )
        
        # Reference outputs (base model without steering)
        with torch.no_grad():
            ref_chosen_logps = self.get_batch_log_probs(
                chosen_outputs["base_logits"],
                chosen_input_ids,
                chosen_attention_mask
            )
            
            ref_rejected_logps = self.get_batch_log_probs(
                rejected_outputs["base_logits"],
                rejected_input_ids,
                rejected_attention_mask
            )
        
        # ========== Compute Losses ==========
        # Main DPO loss
        dpo_loss, reward_margin = self.compute_dpo_loss(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps
        )
        
        # Regularization losses
        steering_reg = self.compute_steering_regularization(chosen_outputs["steering_strength"])
        l2_reg = self.compute_l2_regularization(chosen_outputs["steering_vector"])
        entropy_reg = self.compute_entropy_regularization(
            chosen_outputs["logits"],
            chosen_attention_mask
        )
        
        # Total loss
        total_loss = (
            dpo_loss +
            self.lambda_steering * steering_reg +
            self.lambda_l2 * l2_reg +
            self.lambda_entropy * entropy_reg
        )
        
        # Metrics for logging
        metrics = {
            "loss": total_loss.item(),
            "dpo_loss": dpo_loss.item(),
            "reward_margin": reward_margin.item(),
            "steering_reg": steering_reg.item(),
            "l2_reg": l2_reg.item(),
            "entropy_reg": entropy_reg.item(),
            "avg_steering_strength": chosen_outputs["steering_strength"].mean().item(),
            "std_steering_strength": chosen_outputs["steering_strength"].std().item(),
        }
        
        return total_loss, metrics
    
    def save_checkpoint(self, path: str, metrics: Optional[Dict] = None):
        """Save model checkpoint."""
        
        # Unwrap model if using DDP
        model_to_save = self.accelerator.unwrap_model(self.model)
        
        # Save steering engine state
        checkpoint = {
            "steering_engine_state_dict": model_to_save.steering_engine.state_dict(),
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "args": vars(self.args),
        }
        
        if metrics:
            checkpoint["metrics"] = metrics
        
        # Save checkpoint
        torch.save(checkpoint, path)
        print(f"💾 Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        
        checkpoint = torch.load(path, map_location=self.accelerator.device)
        
        # Load steering engine state
        model_to_load = self.accelerator.unwrap_model(self.model)
        model_to_load.steering_engine.load_state_dict(checkpoint["steering_engine_state_dict"])
        
        # Restore training state
        self.global_step = checkpoint.get("global_step", 0)
        self.best_loss = checkpoint.get("best_loss", float('inf'))
        
        print(f"✅ Loaded checkpoint from {path}")
        print(f"   Resuming from step {self.global_step}")
        
        return checkpoint.get("metrics", {})


def main():
    parser = argparse.ArgumentParser(description="Train GenSteer model")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="argsearch/llama-7b-sft-float32")
    parser.add_argument("--steering_rank", type=int, default=32, help="Rank for steering vector generation")
    parser.add_argument("--max_steering_strength", type=float, default=5.0)
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="Dahoas/full-hh-rlhf")
    parser.add_argument("--max_length", type=int, default=1024)
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=2000)
    
    # Loss weights
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    parser.add_argument("--lambda_l2", type=float, default=0.001)
    parser.add_argument("--lambda_steering", type=float, default=0.01)
    parser.add_argument("--lambda_entropy", type=float, default=0.01)
    
    # Other arguments
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16" if args.bf16 else "no"
    )
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print(f"📚 Loading dataset: {args.dataset_name}")
    full_dataset = load_dataset(args.dataset_name, split="train")
    
    # Split dataset: 90% train, 10% eval
    print("🔄 Splitting dataset (90% train, 10% eval)...")
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Eval samples: {len(eval_dataset)}")
    
    # Prepare datasets
    print("🔧 Processing train dataset...")
    processed_train_dataset = prepare_preference_dataset(
        train_dataset,
        tokenizer,
        max_length=args.max_length
    )
    
    print("🔧 Processing eval dataset...")
    processed_eval_dataset = prepare_preference_dataset(
        eval_dataset,
        tokenizer,
        max_length=args.max_length
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        processed_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    eval_dataloader = DataLoader(
        processed_eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print("🚀 Creating GenSteer model...")
    model = create_gensteer(
        base_model_name=args.model_name,
        steering_rank=args.steering_rank,
        max_steering_strength=args.max_steering_strength,
        device=accelerator.device,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32
    )
    
    # Create optimizer (only for steering engine parameters)
    optimizer = torch.optim.AdamW(
        model.steering_engine.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # Create scheduler
    num_training_steps = len(train_dataloader) * args.num_epochs // args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Prepare for training
    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler
    )
    
    # Create trainer
    trainer = GenSteerTrainer(model, tokenizer, args, accelerator)
    
    # Load checkpoint if resuming
    if args.resume_from_checkpoint:
        trainer.load_checkpoint(args.resume_from_checkpoint)
    
    # Training loop
    print(f"🎯 Starting training for {args.num_epochs} epochs...")
    print(f"   Total steps: {num_training_steps}")
    print(f"   Batch size: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = []
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                # Training step
                loss, metrics = trainer.training_step(batch)
                
                # Backward pass
                accelerator.backward(loss)
                
                # Optimizer step
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                epoch_losses.append(metrics["loss"])
            
            # Update progress bar
            if step % 10 == 0:
                progress_bar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "reward": f"{metrics['reward_margin']:.3f}",
                    "steer": f"{metrics['avg_steering_strength']:.2f}±{metrics['std_steering_strength']:.2f}"
                })
            
            # Save checkpoint
            trainer.global_step += 1
            if trainer.global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"gensteer_step_{trainer.global_step}.pt")
                trainer.save_checkpoint(save_path, metrics)
            
            # Save best model
            if metrics["loss"] < trainer.best_loss:
                trainer.best_loss = metrics["loss"]
                save_path = os.path.join(args.output_dir, "best_gensteer.pt")
                trainer.save_checkpoint(save_path, metrics)
        
        # Epoch summary
        avg_train_loss = np.mean(epoch_losses)
        print(f"📊 Epoch {epoch+1} - Average train loss: {avg_train_loss:.4f}")
        
        # Evaluation
        print(f"🔍 Running evaluation...")
        model.eval()
        eval_losses = []
        
        with torch.no_grad():
            for eval_batch in tqdm(eval_dataloader, desc="Evaluating"):
                loss, eval_metrics = trainer.training_step(eval_batch)
                eval_losses.append(eval_metrics["loss"])
        
        avg_eval_loss = np.mean(eval_losses)
        print(f"📊 Epoch {epoch+1} - Average eval loss: {avg_eval_loss:.4f}")
        print(f"📊 Epoch {epoch+1} - Train/Eval loss: {avg_train_loss:.4f}/{avg_eval_loss:.4f}")
    
    # Save final model
    save_path = os.path.join(args.output_dir, "final_gensteer.pt")
    trainer.save_checkpoint(save_path)
    
    print(f"✅ Training complete! Model saved to {args.output_dir}")
    print(f"🎯 Best loss: {trainer.best_loss:.4f}")


if __name__ == "__main__":
    main()