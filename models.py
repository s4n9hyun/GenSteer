#!/usr/bin/env python3
"""CALM model architecture - Controllable Alignment via Logit Modulation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any, Tuple


class BaseLanguageModel(nn.Module):
    """Frozen base language model wrapper."""
    
    def __init__(self, model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        
        self.device = device
        self.torch_dtype = torch_dtype
        self.model_name = model_name
        
        # Load pre-trained language model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=self.torch_dtype, 
            device_map=None, 
            attn_implementation="eager"
        )
        
        # Freeze all base model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Extract model configuration
        self.config = self.model.config
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        
        self.to(self.device)
        self.eval()
        
        print(f"[CALM] Loaded base model: {model_name}")
        print(f"[CALM] Hidden size: {self.hidden_size}, Vocab size: {self.vocab_size}")
    
    def count_parameters(self) -> int:
        """Count total parameters in base model."""
        return sum(p.numel() for p in self.model.parameters())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get base model information."""
        param_count = self.count_parameters()
        return {
            "model_name": self.model_name,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "parameters": param_count,
            "parameters_B": param_count / 1e9,
        }
    
    def forward(self, input_ids, attention_mask=None, position_ids=None, 
                past_key_values=None, use_cache=False):
        """Forward pass through frozen base model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,  # Required for logit modulation
            return_dict=True
        )
        return outputs


class LogitModulationEngine(nn.Module):
    """Logit Modulation Engine with dynamic vector generation."""
    
    def __init__(self, hidden_size: int = 4096, vocab_size: int = 32000, 
                 bottleneck_dim: int = 32, max_seq_len: int = 2048, 
                 max_modulation_strength: float = 5.0,
                 device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.bottleneck_dim = bottleneck_dim
        self.max_seq_len = max_seq_len
        
        # ============ Logit Modulation Mechanism ============
        # Bottleneck architecture for modulation vector generation
        
        # Compression layer: Project hidden states to bottleneck dimension
        self.compress = nn.Linear(hidden_size, bottleneck_dim, bias=False, dtype=torch_dtype)
        
        # Generation layer: Generate modulation vectors in vocabulary space
        self.generate = nn.Linear(bottleneck_dim, vocab_size, bias=False, dtype=torch_dtype)
        
        # Positional awareness for context-sensitive modulation
        self.pos_encoding = nn.Parameter(
            torch.randn(max_seq_len, hidden_size, dtype=torch_dtype) * 0.01
        )
        
        # Regularization
        self.dropout = nn.Dropout(0.1)
        
        # ============ Automatic Modulation Strength Calibration ============
        # Neural network that learns optimal modulation strength per context
        
        self.strength_calibrator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4, dtype=torch_dtype),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, hidden_size // 16, dtype=torch_dtype),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 16, 1, dtype=torch_dtype),
            nn.Sigmoid()  # Outputs normalized strength in [0, 1]
        )
        
        # Maximum modulation strength (expanded range for better exploration)
        self.max_modulation_strength = max_modulation_strength
        
        # Initialize parameters optimally
        self._initialize_parameters()
        
        self.to(device)
    
    def _initialize_parameters(self):
        """Initialize LME parameters for optimal training."""
        with torch.no_grad():
            # Generative mechanism initialization
            # Compress: Normal initialization scaled by input dimension
            nn.init.normal_(self.compress.weight, std=1.0 / self.compress.in_features)
            # Generate: Zero initialization for minimal initial modulation
            nn.init.zeros_(self.generate.weight)
            
            # Positional encoding: Small random values
            nn.init.normal_(self.pos_encoding, mean=0.0, std=0.01)
            
            # Strength calibrator: Xavier initialization for stable gradients
            for layer in self.strength_calibrator:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def adaptive_pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Resize hidden states to match expected dimensions."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        if hidden_dim == self.hidden_size:
            return hidden_states
        elif hidden_dim < self.hidden_size:
            # Upsample for smaller hidden dimensions
            if not hasattr(self, '_upsample'):
                self._upsample = nn.Linear(hidden_dim, self.hidden_size, 
                                          dtype=hidden_states.dtype).to(hidden_states.device)
                nn.init.xavier_uniform_(self._upsample.weight)
            return self._upsample(hidden_states)
        else:
            # Downsample for larger hidden dimensions
            reshaped = hidden_states.view(batch_size * seq_len, 1, hidden_dim)
            pooled = F.adaptive_avg_pool1d(reshaped, self.hidden_size)
            return pooled.squeeze(1).view(batch_size, seq_len, self.hidden_size)
    
    def count_parameters(self) -> int:
        """Count trainable parameters in LME."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get modulation engine information."""
        param_count = self.count_parameters()
        calibrator_params = sum(p.numel() for name, p in self.named_parameters() 
                               if 'strength_calibrator' in name and p.requires_grad)
        modulation_params = sum(p.numel() for name, p in self.named_parameters() 
                            if ('compress' in name or 'generate' in name) and p.requires_grad)
        
        return {
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "bottleneck_dim": self.bottleneck_dim,
            "parameters": param_count,
            "parameters_M": param_count / 1e6,
            "calibrator_parameters": calibrator_params,
            "modulation_parameters": modulation_params,
            "max_modulation_strength": self.max_modulation_strength,
        }
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """
        Generate modulation vectors and calibrate modulation strength.
        
        Args:
            hidden_states: Last hidden states from base model [B, L, H]
            
        Returns:
            modulation_vector: Generated modulation adjustments [B, L, V]
            info: Dictionary with generation info
            modulation_strength: Calibrated modulation strength [B, 1]
        """
        
        # Adaptive pooling for dimension compatibility  
        pooled_states = self.adaptive_pool(hidden_states)  # [B, L, H]
        batch_size, seq_len, hidden_size = pooled_states.shape
        
        # ============ Logit Modulation Vector Creation ============
        
        # Add positional information for context awareness
        if seq_len <= self.max_seq_len:
            pos_encoding = self.pos_encoding[:seq_len, :].unsqueeze(0)
            contextualized = self.dropout(pooled_states + pos_encoding)
        else:
            contextualized = self.dropout(pooled_states)
        
        # Generate modulation vectors through bottleneck
        # Step 1: Compress to bottleneck dimension [B, L, H] → [B, L, D]
        compressed = self.compress(contextualized)
        
        # Step 2: Generate modulation vectors [B, L, D] → [B, L, V]
        modulation_vector = self.generate(compressed)
        
        # ============ Automatic Modulation Strength Calibration ============
        
        # Compute global context representation
        context_vector = pooled_states.mean(dim=1)  # [B, H]
        
        # Calibrate modulation strength based on context
        normalized_strength = self.strength_calibrator(context_vector)  # [B, 1] in [0, 1]
        modulation_strength = normalized_strength * self.max_modulation_strength  # [B, 1] in [0, 5.0]
        
        # Prepare information dictionary
        info = {
            "normalized_strength": normalized_strength,
            "bottleneck_dim": self.bottleneck_dim,
            "compression_ratio": hidden_size / self.bottleneck_dim,
        }
        
        return modulation_vector, info, modulation_strength


class CALMModel(nn.Module):
    """Complete CALM system with base model and modulation engine."""
    
    def __init__(self, base_model: BaseLanguageModel, modulation_engine: LogitModulationEngine):
        super().__init__()
        
        self.base_model = base_model
        self.modulation_engine = modulation_engine
        
        # Verify vocabulary compatibility
        assert base_model.vocab_size == modulation_engine.vocab_size, \
            f"Vocabulary mismatch: base={base_model.vocab_size}, modulation={modulation_engine.vocab_size}"
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        base_info = self.base_model.get_model_info()
        engine_info = self.modulation_engine.get_engine_info()
        
        return {
            "system": "CALM v1.0",
            "base_model": base_info,
            "modulation_engine": engine_info,
            "compatibility": {
                "adaptive_pooling_needed": base_info["hidden_size"] != engine_info["hidden_size"],
                "pooling_ratio": base_info["hidden_size"] / engine_info["hidden_size"],
                "vocab_compatible": base_info["vocab_size"] == engine_info["vocab_size"],
            },
            "architecture": "Controllable Alignment via Logit Modulation"
        }
    
    def forward(self, input_ids, attention_mask=None, position_ids=None, 
                past_key_values=None, use_cache=False, return_components=False):
        """
        Forward pass with automatic logit modulation.
        
        The model automatically determines optimal modulation strength and
        generates appropriate modulation vectors for the input context.
        """
        
        # Get base model outputs
        base_outputs = self.base_model(
            input_ids, attention_mask, position_ids, 
            past_key_values, use_cache
        )
        
        # Extract hidden states for modulation
        last_hidden_states = base_outputs.hidden_states[-1]
        
        # Generate modulation vectors with automatic calibration
        modulation_vector, modulation_info, modulation_strength = self.modulation_engine(last_hidden_states)
        
        # Apply calibrated modulation to base model logits
        # modulation_strength: [B, 1] -> [B, 1, 1] for broadcasting
        strength_expanded = modulation_strength.unsqueeze(-1)
        modulated_logits = base_outputs.logits + strength_expanded * modulation_vector
        
        # Prepare outputs
        outputs = {
            "logits": modulated_logits,
            "past_key_values": base_outputs.past_key_values if use_cache else None,
        }
        
        if return_components:
            outputs.update({
                "base_logits": base_outputs.logits,
                "modulation_vector": modulation_vector,
                "modulation_strength": modulation_strength,
                "modulation_info": modulation_info,
            })
        
        return outputs
    
    def generate_with_modulation(self, input_ids, max_length=128, temperature=1.0,
                              do_sample=False, top_p=1.0, attention_mask=None):
        """
        Generate text with automatic modulation calibration.
        
        The modulation strength is automatically determined for each token,
        adapting to the evolving context during generation.
        """
        
        if input_ids.numel() == 0:
            return input_ids, {"modulation_strengths": [], "avg_modulation_strength": 0.0}
        
        self.eval()
        generated = input_ids.clone()
        modulation_strengths = []
        modulation_norms = []
        past_key_values = None
        
        with torch.no_grad():
            for step in range(max_length):
                # Prepare inputs
                if step == 0:
                    current_input_ids = generated
                    current_attention_mask = attention_mask
                else:
                    current_input_ids = generated[:, -1:]
                    current_attention_mask = None
                
                # Forward pass with automatic modulation
                outputs = self.forward(
                    current_input_ids, current_attention_mask, 
                    past_key_values=past_key_values, use_cache=True,
                    return_components=True
                )
                
                past_key_values = outputs.get("past_key_values")
                
                # Record modulation strength for this step
                modulation_strength = outputs["modulation_strength"].item()
                modulation_strengths.append(modulation_strength)
                
                # Sample next token
                logits = outputs["logits"][:, -1, :]
                
                if temperature > 0:
                    logits = logits / temperature
                
                if do_sample:
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        logits.scatter_(1, indices_to_remove, float('-inf'))
                    
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Track modulation vector norm
                if "modulation_vector" in outputs:
                    modulation_norm = outputs["modulation_vector"][:, -1, :].norm(dim=-1).mean().item()
                    modulation_norms.append(modulation_norm)
                
                # Stop on EOS token
                if next_token.item() == self.base_model.model.config.eos_token_id:
                    break
        
        # Compile generation statistics
        stats = {
            "modulation_strengths": modulation_strengths,
            "avg_modulation_strength": sum(modulation_strengths) / len(modulation_strengths) if modulation_strengths else 0.0,
            "modulation_norms": modulation_norms,
            "avg_modulation_norm": sum(modulation_norms) / len(modulation_norms) if modulation_norms else 0.0,
            "bottleneck_dim": self.modulation_engine.bottleneck_dim,
        }
        
        return generated, stats


def create_calm_model(base_model_name: str = "argsearch/llama-7b-sft-float32", 
                    device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16,
                    hidden_size: int = 4096, vocab_size: Optional[int] = None, 
                    bottleneck_dim: int = 32, max_modulation_strength: float = 5.0) -> CALMModel:
    """
    Factory function to create a CALM model.
    
    Args:
        base_model_name: Name or path of the base language model
        device: Device to run the model on
        torch_dtype: Data type for model weights
        hidden_size: Hidden dimension of the model
        vocab_size: Vocabulary size (auto-detected if None)
        bottleneck_dim: Bottleneck dimension for modulation generation
        max_modulation_strength: Maximum modulation strength
        
    Returns:
        Complete CALM model ready for training or inference
    """
    
    # Load base language model
    base_model = BaseLanguageModel(base_model_name, device, torch_dtype)
    
    # Auto-detect vocabulary size if not provided
    if vocab_size is None:
        vocab_size = base_model.vocab_size
    
    # Create Logit Modulation Engine
    modulation_engine = LogitModulationEngine(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        bottleneck_dim=bottleneck_dim,
        max_modulation_strength=max_modulation_strength,
        device=device,
        torch_dtype=torch_dtype
    )
    
    # Combine into complete CALM system
    calm_model = CALMModel(base_model, modulation_engine)
    
    print(f"[CALM] Created model with bottleneck dimension {bottleneck_dim}")
    print(f"[CALM] Modulation parameters: {modulation_engine.count_parameters()/1e6:.1f}M")
    print(f"[CALM] Max modulation strength: {modulation_engine.max_modulation_strength}")
    
    return calm_model


# Backward compatibility alias
def create_gensteer(*args, **kwargs):
    """Backward compatibility alias for create_calm_model."""
    return create_calm_model(*args, **kwargs)