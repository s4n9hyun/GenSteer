#!/usr/bin/env python3
"""CALM inference utilities."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import List, Dict, Optional, Union, Tuple
import json
import numpy as np
from pathlib import Path

from models import create_calm_model


class CALMInference:
    """Inference engine for CALM models."""
    
    def __init__(
        self,
        checkpoint_path: str,
        base_model_name: str = "argsearch/llama-7b-sft-float32",
        device: str = "cuda",
        torch_dtype = torch.bfloat16,
        bottleneck_dim: int = 32,
        max_modulation_strength: float = 5.0
    ):
        """Initialize CALM inference engine."""
        
        self.device = device
        self.torch_dtype = torch_dtype
        self.checkpoint_path = checkpoint_path
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create model
        print(f"🚀 Loading CALM model from {checkpoint_path}")
        self.model = create_calm_model(
            base_model_name=base_model_name,
            bottleneck_dim=bottleneck_dim,
            max_modulation_strength=max_modulation_strength,
            device=device,
            torch_dtype=torch_dtype
        )
        
        # Load checkpoint
        self._load_checkpoint(checkpoint_path)
        
        # Set to evaluation mode
        self.model.eval()
        
        print("✅ CALM inference engine ready!")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load modulation engine state - handle both old and new checkpoint formats
        if "modulation_engine_state_dict" in checkpoint:
            self.model.modulation_engine.load_state_dict(checkpoint["modulation_engine_state_dict"])
        elif "alignment_model" in checkpoint:
            # New format from training script
            self.model.modulation_engine.load_state_dict(checkpoint["alignment_model"])
        elif "steering_engine_state_dict" in checkpoint:
            # Legacy format compatibility
            self.model.modulation_engine.load_state_dict(checkpoint["steering_engine_state_dict"])
        else:
            raise KeyError(f"Cannot find modulation engine weights in checkpoint. Available keys: {checkpoint.keys()}")
        
        # Load training info if available
        self.training_info = {
            "global_step": checkpoint.get("global_step", 0),
            "best_loss": checkpoint.get("best_loss", float('inf')),
            "metrics": checkpoint.get("metrics", {}),
        }
        
        print(f"✅ Loaded checkpoint from step {self.training_info['global_step']}")
        if "metrics" in checkpoint:
            metrics = checkpoint["metrics"]
            if "avg_modulation_strength" in metrics:
                print(f"   Avg modulation: {metrics['avg_modulation_strength']:.2f}±{metrics.get('std_modulation_strength', 0):.2f}")
            elif "avg_steering_strength" in metrics:
                # Legacy compatibility
                print(f"   Avg modulation: {metrics['avg_steering_strength']:.2f}±{metrics.get('std_steering_strength', 0):.2f}")
    
    def generate_response(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        do_sample: bool = True,
        return_steering_info: bool = False
    ) -> Union[str, Tuple[str, Dict]]:
        """
        Generate a response with automatic modulation.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            return_steering_info: Whether to return modulation statistics
        
        Returns:
            Generated response (and optionally modulation info)
        """
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Generate with modulation
        with torch.no_grad():
            generated_ids, stats = self.model.generate_with_modulation(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample
            )
        
        # Decode response
        input_length = inputs["input_ids"].shape[1]
        response_ids = generated_ids[:, input_length:]
        response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
        
        if return_steering_info:
            return response, stats
        else:
            return response
    
    def batch_generate(
        self,
        prompts: List[str],
        max_length: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        do_sample: bool = True,
        batch_size: int = 8
    ) -> List[Dict]:
        """
        Generate responses for multiple prompts in batches.
        
        Args:
            prompts: List of input prompts
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            batch_size: Batch size for processing
        
        Returns:
            List of response dictionaries with modulation info
        """
        
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = []
            
            for prompt in batch_prompts:
                response, modulation_info = self.generate_response(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    return_steering_info=True
                )
                
                batch_results.append({
                    "prompt": prompt,
                    "response": response,
                    "modulation_info": modulation_info
                })
            
            results.extend(batch_results)
        
        return results
    
    def compare_with_base(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Dict:
        """
        Compare CALM response with base model response.
        
        Returns:
            Dictionary with both responses and modulation analysis
        """
        
        # Generate with CALM
        calm_response, modulation_info = self.generate_response(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            return_steering_info=True
        )
        
        # Generate with base model only
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        with torch.no_grad():
            # Use base model directly
            base_outputs = self.model.base_model(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                use_cache=True
            )
            
            # Generate with base model
            base_generated = self._generate_with_logits(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                logits=base_outputs.logits,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample
            )
        
        # Decode base response
        input_length = inputs["input_ids"].shape[1]
        base_response_ids = base_generated[:, input_length:]
        base_response = self.tokenizer.decode(base_response_ids[0], skip_special_tokens=True)
        
        return {
            "prompt": prompt,
            "calm_response": calm_response,
            "base_response": base_response,
            "modulation_info": modulation_info,
            "modulation_active": modulation_info.get("avg_modulation_strength", 0) > 0.1
        }
    
    def _generate_with_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        logits: torch.Tensor,
        max_length: int,
        temperature: float,
        top_p: float,
        do_sample: bool
    ) -> torch.Tensor:
        """Helper function for base model generation."""
        
        generated = input_ids.clone()
        past_key_values = None
        
        for _ in range(max_length):
            # Get next token logits
            next_logits = logits[:, -1, :]
            
            if temperature > 0:
                next_logits = next_logits / temperature
            
            if do_sample:
                if top_p < 1.0:
                    # Top-p sampling
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_logits.scatter_(1, indices_to_remove, float('-inf'))
                
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            
            # Update generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop on EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Continue generation (simplified - would need full forward pass)
            break
        
        return generated
    
    def analyze_modulation_patterns(
        self,
        prompts: List[str],
        max_length: int = 256
    ) -> Dict:
        """
        Analyze modulation patterns across multiple prompts.
        
        Returns:
            Analysis of modulation behavior patterns
        """
        
        results = self.batch_generate(prompts, max_length=max_length)
        
        # Extract modulation strengths
        modulation_strengths = []
        modulation_variances = []
        
        for result in results:
            info = result["modulation_info"]
            if "modulation_strengths" in info:
                strengths = info["modulation_strengths"]
                modulation_strengths.extend(strengths)
                if len(strengths) > 1:
                    modulation_variances.append(np.var(strengths))
            elif "learned_steering_strengths" in info:
                # Legacy compatibility
                strengths = info["learned_steering_strengths"]
                modulation_strengths.extend(strengths)
                if len(strengths) > 1:
                    modulation_variances.append(np.var(strengths))
        
        if not modulation_strengths:
            return {"error": "No modulation information available"}
        
        analysis = {
            "total_generations": len(results),
            "total_tokens": len(modulation_strengths),
            "modulation_statistics": {
                "mean": np.mean(modulation_strengths),
                "std": np.std(modulation_strengths),
                "min": np.min(modulation_strengths),
                "max": np.max(modulation_strengths),
                "median": np.median(modulation_strengths),
            },
            "variance_statistics": {
                "mean_variance": np.mean(modulation_variances) if modulation_variances else 0,
                "adaptive_modulation": np.mean(modulation_variances) > 0.1 if modulation_variances else False,
            },
            "utilization": {
                "active_modulation_ratio": np.mean(np.array(modulation_strengths) > 0.5),
                "high_modulation_ratio": np.mean(np.array(modulation_strengths) > 2.0),
            }
        }
        
        return analysis
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information."""
        
        system_info = self.model.get_system_info()
        
        info = {
            "model_type": "CALM",
            "checkpoint_path": self.checkpoint_path,
            "training_info": self.training_info,
            "system_info": system_info,
            "architecture": "Controllable Alignment via Logit Modulation",
        }
        
        return info


def load_calm_inference(
    checkpoint_path: str,
    base_model_name: str = "argsearch/llama-7b-sft-float32",
    device: str = "cuda",
    **kwargs
) -> CALMInference:
    """
    Convenience function to load CALM inference engine.
    
    Args:
        checkpoint_path: Path to trained checkpoint
        base_model_name: Base model identifier
        device: Device for inference
        **kwargs: Additional arguments for CALMInference
    
    Returns:
        Ready-to-use CALMInference instance
    """
    
    return CALMInference(
        checkpoint_path=checkpoint_path,
        base_model_name=base_model_name,
        device=device,
        **kwargs
    )


# Backward compatibility alias
def load_gensteer_inference(*args, **kwargs):
    """Backward compatibility alias for load_calm_inference."""
    return load_calm_inference(*args, **kwargs)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="CALM Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default="Hello, how can I help you today?")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--compare", action="store_true", help="Compare with base model")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = load_calm_inference(args.checkpoint)
    
    print(f"🎯 Generating response for: {args.prompt}")
    
    if args.compare:
        # Compare with base model
        result = engine.compare_with_base(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        print("\n🤖 CALM Response:")
        print(result["calm_response"])
        print(f"\n🔧 Modulation: {result['modulation_info']['avg_modulation_strength']:.2f}")
        
        print("\n📝 Base Model Response:")
        print(result["base_response"])
        
        print(f"\n🎛️  Modulation Active: {result['modulation_active']}")
        
    else:
        # Generate with CALM
        response, modulation_info = engine.generate_response(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            return_steering_info=True
        )
        
        print(f"\n🤖 Response: {response}")
        print(f"🔧 Avg Modulation: {modulation_info.get('avg_modulation_strength', 0):.2f}")