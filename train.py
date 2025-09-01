#!/usr/bin/env python3
"""
CALM Training Script
Paper: CALM: Controllable Alignment via Logit Modulation
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from typing import Dict, Tuple
from tqdm import tqdm
import numpy as np

from models import create_calm_model
from data import PreferenceDataset, load_preference_data


class AlignmentLoss(nn.Module):
    """Alignment loss function for CALM."""
    
    def __init__(self, beta=0.1, lambda_l2=0.001, lambda_strength_variance=0.01, lambda_entropy=0.01):
        super().__init__()
        self.beta = beta
        self.lambda_l2 = lambda_l2
        self.lambda_strength_variance = lambda_strength_variance  # Regularization for modulation strength variance
        self.lambda_entropy = lambda_entropy  # Entropy regularization to prevent repetition
    
    def compute_preference_loss(self, logits_chosen, logits_rejected, labels_chosen, labels_rejected, mask_chosen, mask_rejected):
        """DPO preference loss."""
        
        log_probs_chosen = self._get_log_probs(logits_chosen, labels_chosen)
        log_probs_rejected = self._get_log_probs(logits_rejected, labels_rejected)
        
        # Shift masks
        mask_chosen_shifted = mask_chosen[:, 1:].contiguous()
        mask_rejected_shifted = mask_rejected[:, 1:].contiguous()
        
        # Average log probs
        log_probs_chosen = (log_probs_chosen * mask_chosen_shifted).sum(dim=1) / mask_chosen_shifted.sum(dim=1)
        log_probs_rejected = (log_probs_rejected * mask_rejected_shifted).sum(dim=1) / mask_rejected_shifted.sum(dim=1)
        
        # DPO loss
        preference_loss = -F.logsigmoid(self.beta * (log_probs_chosen - log_probs_rejected)).mean()
        return preference_loss
    
    def compute_l2_regularization(self, modulation_vector, attention_mask):
        """L2 regularization for modulation vectors."""
        l2_norms = modulation_vector.norm(dim=-1)
        masked_l2_penalty = l2_norms * attention_mask
        return masked_l2_penalty.sum() / (attention_mask.sum() + 1e-8)
    
    def compute_strength_variance_regularization(self, modulation_strength):
        """Regularization for modulation strength to encourage exploration."""
        if modulation_strength.numel() <= 1:
            return torch.tensor(0.0, device=modulation_strength.device)
        
        # Encourage variance in modulation values to promote exploration
        strength_variance = torch.var(modulation_strength)
        
        # Return negative variance as loss (minimize negative variance = maximize variance)
        return -strength_variance
    
    def compute_entropy_regularization(self, logits, attention_mask):
        """Entropy regularization to discourage repetition."""
        # Flatten logits and mask for easier processing
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        mask_flat = attention_mask.reshape(-1)
        
        # Only compute entropy for non-padded tokens
        valid_indices = mask_flat.bool()
        if not valid_indices.any():
            return torch.tensor(0.0, device=logits.device)
        
        logits_valid = logits_flat[valid_indices]
        
        # Compute probability distribution
        probs = F.softmax(logits_valid, dim=-1)
        log_probs = F.log_softmax(logits_valid, dim=-1)
        
        # Compute entropy: H(p) = -sum(p * log(p))
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # Return negative entropy as loss (minimize negative entropy = maximize entropy)
        return -entropy.mean()
    
    def _get_log_probs(self, logits, labels):
        """Get log probabilities for labels."""
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        labels_shifted = labels[:, 1:].unsqueeze(-1)
        return log_probs.gather(dim=-1, index=labels_shifted).squeeze(-1)
    
    def forward(self, model_outputs_chosen, model_outputs_rejected, batch):
        """
        Compute total alignment loss with all regularization terms.
        
        Args:
            model_outputs_chosen: Model outputs for chosen responses
            model_outputs_rejected: Model outputs for rejected responses  
            batch: Batch data with input_ids, attention_mask, etc.
        """
        
        # Extract components
        logits_chosen = model_outputs_chosen["logits"]
        logits_rejected = model_outputs_rejected["logits"]
        modulation_vector_chosen = model_outputs_chosen.get("modulation_vector")
        modulation_vector_rejected = model_outputs_rejected.get("modulation_vector")
        modulation_strength_chosen = model_outputs_chosen.get("modulation_strength")
        modulation_strength_rejected = model_outputs_rejected.get("modulation_strength")
        
        # Get labels and masks
        labels_chosen = batch["chosen_input_ids"]
        labels_rejected = batch["rejected_input_ids"]
        mask_chosen = batch["chosen_attention_mask"]
        mask_rejected = batch["rejected_attention_mask"]
        
        # 1. Main preference loss (DPO)
        preference_loss = self.compute_preference_loss(
            logits_chosen, logits_rejected,
            labels_chosen, labels_rejected,
            mask_chosen, mask_rejected
        )
        
        total_loss = preference_loss
        loss_components = {"preference_loss": preference_loss.item()}
        
        # 2. L2 regularization on modulation vectors
        if modulation_vector_chosen is not None and self.lambda_l2 > 0:
            l2_loss_chosen = self.compute_l2_regularization(modulation_vector_chosen, mask_chosen)
            l2_loss_rejected = self.compute_l2_regularization(modulation_vector_rejected, mask_rejected)
            l2_loss = (l2_loss_chosen + l2_loss_rejected) / 2
            total_loss += self.lambda_l2 * l2_loss
            loss_components["l2_loss"] = l2_loss.item()
        
        # 3. Strength variance regularization to encourage exploration
        if modulation_strength_chosen is not None and self.lambda_strength_variance > 0:
            strength_combined = torch.cat([modulation_strength_chosen.flatten(), modulation_strength_rejected.flatten()])
            strength_reg_loss = self.compute_strength_variance_regularization(strength_combined)
            total_loss += self.lambda_strength_variance * strength_reg_loss
            loss_components["strength_variance_loss"] = strength_reg_loss.item()
        
        # 4. Entropy regularization to prevent repetitive generation
        if self.lambda_entropy > 0:
            entropy_loss_chosen = self.compute_entropy_regularization(logits_chosen, mask_chosen)
            entropy_loss_rejected = self.compute_entropy_regularization(logits_rejected, mask_rejected)
            entropy_loss = (entropy_loss_chosen + entropy_loss_rejected) / 2
            total_loss += self.lambda_entropy * entropy_loss
            loss_components["entropy_loss"] = entropy_loss.item()
        
        loss_components["total_loss"] = total_loss.item()
        
        return total_loss, loss_components


class PreferenceDataset:
    """Preference dataset with real-time tokenization."""
    
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")
        
        chosen_tokens = self.tokenizer(
            chosen,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        rejected_tokens = self.tokenizer(
            rejected,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids_chosen": chosen_tokens["input_ids"].squeeze(0),
            "attention_mask_chosen": chosen_tokens["attention_mask"].squeeze(0),
            "labels_chosen": chosen_tokens["input_ids"].squeeze(0),
            "input_ids_rejected": rejected_tokens["input_ids"].squeeze(0),
            "attention_mask_rejected": rejected_tokens["attention_mask"].squeeze(0),
            "labels_rejected": rejected_tokens["input_ids"].squeeze(0)
        }




def main():
    parser = argparse.ArgumentParser(description="Train CALM model")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="argsearch/llama-7b-sft-float32")
    parser.add_argument("--bottleneck_dim", type=int, default=32, help="Bottleneck dimension for modulation generation")
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="Dahoas/full-hh-rlhf")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use (auto-converts to train_prefs for UltraFeedback)")
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
    parser.add_argument("--lambda_strength_variance", type=float, default=0.01)
    # Backward compatibility
    parser.add_argument("--lambda_steering", type=float, help="Alias for lambda_strength_variance (deprecated)")  
    parser.add_argument("--lambda_entropy", type=float, default=0.01)
    
    # Other arguments
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    args = parser.parse_args()
    
    # Handle backward compatibility
    if args.lambda_steering is not None:
        args.lambda_strength_variance = args.lambda_steering
        print(f"⚠️ Warning: --lambda_steering is deprecated, use --lambda_strength_variance instead")
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16" if args.bf16 else "no"
    )
    
    print("🚀 Training CALM - Controllable Alignment via Logit Modulation")
    print("📚 Paper: CALM: Controllable Alignment via Logit Modulation")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load preference dataset
    print(f"Loading dataset: {args.dataset_name} (split: {args.dataset_split})")
    preference_data = load_preference_data(args.dataset_name, split=args.dataset_split)
    
    # Split into train and eval (90/10)
    split_idx = int(0.9 * len(preference_data))
    train_data = preference_data[:split_idx]
    eval_data = preference_data[split_idx:]
    
    # Create datasets
    train_dataset = PreferenceDataset(train_data, tokenizer, args.max_length)
    eval_dataset = PreferenceDataset(eval_data, tokenizer, args.max_length)
    
    print(f"Loaded {len(train_dataset)} training samples, {len(eval_dataset)} eval samples")
    
    # Define collate function
    def collate_fn(batch):
        """Collate function for preference data."""
        return {
            "chosen_input_ids": torch.stack([item["input_ids_chosen"] for item in batch]),
            "chosen_attention_mask": torch.stack([item["attention_mask_chosen"] for item in batch]),
            "rejected_input_ids": torch.stack([item["input_ids_rejected"] for item in batch]),
            "rejected_attention_mask": torch.stack([item["attention_mask_rejected"] for item in batch]),
        }
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    # Create CALM model
    print(f"Creating CALM model with bottleneck dim {args.bottleneck_dim}...")
    model = create_calm_model(
        base_model_name=args.model_name,
        bottleneck_dim=args.bottleneck_dim,
        max_modulation_strength=5.0,  # Fixed default value
        device=accelerator.device,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32
    )
    
    # Print model info
    system_info = model.get_system_info()
    print(f"\n📊 CALM Model Information:")
    print(f"  System: {system_info['system']}")
    print(f"  Architecture: {system_info['architecture']}")
    print(f"  Base model: {args.model_name}")
    print(f"  Modulation parameters: {system_info['modulation_engine']['parameters_M']:.1f}M")
    print(f"  Bottleneck dim: {system_info['modulation_engine']['bottleneck_dim']}")
    print(f"  Max modulation strength: {system_info['modulation_engine']['max_modulation_strength']}")
    
    # Loss function with all regularizations
    loss_fn = AlignmentLoss(
        beta=args.beta,
        lambda_l2=args.lambda_l2,
        lambda_strength_variance=args.lambda_strength_variance,
        lambda_entropy=args.lambda_entropy
    )
    
    # Optimizer (only modulation engine parameters)
    model.alignment_model = model.modulation_engine
    optimizer = torch.optim.AdamW(
        model.alignment_model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    num_training_steps = len(train_loader) * args.num_epochs // args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Prepare with accelerator
    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )
    
    # Resume from checkpoint if specified
    # Initialize global_step before checkpoint loading
    global_step = 0
    
    if args.resume_from_checkpoint:
        print(f"🔄 Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=accelerator.device, weights_only=False)
        
        # Handle both wrapped and unwrapped models
        if hasattr(model, 'module'):
            alignment_model = model.module.alignment_model
        else:
            alignment_model = model.alignment_model
            
        alignment_model.load_state_dict(checkpoint['alignment_model'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
            print(f"📍 Resuming from step {global_step}")
        print("✅ Checkpoint loaded successfully")
    
    # Training loop
    model.train()
    best_eval_loss = float('inf')
    
    # Calculate starting epoch and step when resuming
    steps_per_epoch = len(train_loader)
    start_epoch = global_step // steps_per_epoch if global_step > 0 else 0
    start_step = global_step % steps_per_epoch if global_step > 0 else 0
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n🎯 Epoch {epoch + 1}/{args.num_epochs}")
        
        epoch_losses = []
        progress_bar = tqdm(train_loader, desc=f"Training", disable=not accelerator.is_main_process)
        
        for step, batch in enumerate(progress_bar):
            # Skip already completed steps when resuming
            if epoch == start_epoch and step < start_step:
                continue
            with accelerator.accumulate(model):
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
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    # Handle both wrapped and unwrapped models
                    if hasattr(model, 'module'):
                        alignment_model = model.module.alignment_model
                    else:
                        alignment_model = model.alignment_model
                    accelerator.clip_grad_norm_(alignment_model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                if accelerator.sync_gradients:
                    global_step += 1
                
                epoch_losses.append(loss_components["total_loss"])
                
                # Update progress bar
                if accelerator.is_main_process:
                    # Get modulation statistics
                    chosen_strength = chosen_outputs["modulation_strength"].detach().cpu().numpy()
                    rejected_strength = rejected_outputs["modulation_strength"].detach().cpu().numpy()
                    all_strengths = np.concatenate([chosen_strength.flatten(), rejected_strength.flatten()])
                    
                    progress_bar.set_postfix({
                        'loss': f"{loss_components['total_loss']:.4f}",
                        'mod_strength_mean': f"{all_strengths.mean():.3f}",
                        'mod_strength_std': f"{all_strengths.std():.3f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                    })
                
                # Periodic evaluation and saving (skip step 0)
                if global_step > 0 and global_step % args.eval_steps == 0 and accelerator.is_main_process:
                    print(f"\n🔍 Evaluating at step {global_step}")
                    
                    model.eval()
                    eval_losses = []
                    
                    with torch.no_grad():
                        for eval_batch in eval_loader:
                            eval_chosen_outputs = model(
                                input_ids=eval_batch["chosen_input_ids"],
                                attention_mask=eval_batch["chosen_attention_mask"],
                                return_components=True
                            )
                            eval_rejected_outputs = model(
                                input_ids=eval_batch["rejected_input_ids"],
                                attention_mask=eval_batch["rejected_attention_mask"],
                                return_components=True
                            )
                            eval_loss, eval_loss_components = loss_fn(eval_chosen_outputs, eval_rejected_outputs, eval_batch)
                            eval_losses.append(eval_loss_components["total_loss"])
                    
                    avg_eval_loss = np.mean(eval_losses)
                    print(f"📊 Evaluation - Loss: {avg_eval_loss:.4f}")
                    
                    # Save checkpoint if best
                    if avg_eval_loss < best_eval_loss:
                        best_eval_loss = avg_eval_loss
                        
                        # Save model
                        if hasattr(model, 'module'):
                            model_to_save = model.module.alignment_model
                        else:
                            model_to_save = model.alignment_model
                        
                        checkpoint = {
                            'alignment_model': model_to_save.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'global_step': global_step,
                            'eval_loss': avg_eval_loss,
                            'args': vars(args)
                        }
                        
                        save_path = os.path.join(args.output_dir, "best_calm.pt")
                        torch.save(checkpoint, save_path)
                        print(f"💾 Saved best model to {save_path}")
                    
                    model.train()
                
                # Regular checkpoint saving
                if global_step % args.save_steps == 0 and accelerator.is_main_process:
                    # Save model
                    if hasattr(model, 'module'):
                        model_to_save = model.module.alignment_model
                    else:
                        model_to_save = model.alignment_model
                    
                    checkpoint = {
                        'alignment_model': model_to_save.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'global_step': global_step,
                        'args': vars(args)
                    }
                    
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                    torch.save(checkpoint, save_path)
                    print(f"💾 Saved checkpoint to {save_path}")
        
        # End of epoch summary
        avg_epoch_loss = np.mean(epoch_losses)
        print(f"📊 Epoch {epoch + 1} completed - Average loss: {avg_epoch_loss:.4f}")
    
    # Save final model after training completion
    if accelerator.is_main_process:
        print("💾 Saving final model...")
        
        # Get the model to save
        if hasattr(model, 'module'):
            model_to_save = model.module.alignment_model
        else:
            model_to_save = model.alignment_model
        
        # Save final checkpoint
        final_checkpoint = {
            'alignment_model': model_to_save.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'global_step': global_step,
            'args': vars(args)
        }
        
        # Save as final.pt
        final_path = os.path.join(args.output_dir, "final.pt")
        torch.save(final_checkpoint, final_path)
        print(f"💾 Saved final model to {final_path}")
        
        # If no best model was saved (no eval), use final as best
        best_path = os.path.join(args.output_dir, "best_calm.pt")
        if not os.path.exists(best_path):
            torch.save(final_checkpoint, best_path)
            print(f"💾 Saved final model as best to {best_path}")
    
    print(f"✅ Training complete! Best eval loss: {best_eval_loss:.4f}")
    print(f"🎯 Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()