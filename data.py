#!/usr/bin/env python3
"""
GenSteer Data Processing Utilities
Paper: GenSteer: A Generative Steering Engine for Autonomous Test-Time Alignment

Optimized data processing for preference learning and evaluation datasets.
"""

import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import json
import os
from pathlib import Path
import warnings
from tqdm import tqdm


class PreferenceDataProcessor:
    """Optimized processor for preference datasets used in GenSteer training."""
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
        truncation_strategy: str = "right",
        add_special_tokens: bool = True
    ):
        """
        Initialize preference data processor.
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            truncation_strategy: How to truncate long sequences ("right", "left")
            add_special_tokens: Whether to add special tokens
        """
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.add_special_tokens = add_special_tokens
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            warnings.warn("Pad token was None, set to eos_token")
    
    def process_single_example(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process a single preference example.
        
        Args:
            example: Dictionary with 'chosen' and 'rejected' keys
        
        Returns:
            Dictionary with tokenized chosen and rejected sequences
        """
        
        chosen = example.get("chosen", "")
        rejected = example.get("rejected", "")
        
        # Handle different input formats
        if isinstance(chosen, list):
            chosen = self.tokenizer.eos_token.join(chosen)
        if isinstance(rejected, list):
            rejected = self.tokenizer.eos_token.join(rejected)
        
        # Tokenize chosen sequence
        chosen_tokens = self.tokenizer(
            chosen,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=self.add_special_tokens
        )
        
        # Tokenize rejected sequence
        rejected_tokens = self.tokenizer(
            rejected,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=self.add_special_tokens
        )
        
        return {
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(0),
        }
    
    def process_dataset(
        self,
        dataset: Union[Dataset, List[Dict]],
        num_proc: int = 4,
        cache_dir: Optional[str] = None
    ) -> Dataset:
        """
        Process entire preference dataset.
        
        Args:
            dataset: Input dataset or list of examples
            num_proc: Number of processes for parallel processing
            cache_dir: Directory for caching processed data
        
        Returns:
            Processed dataset ready for training
        """
        
        if isinstance(dataset, list):
            dataset = Dataset.from_list(dataset)
        
        # Process examples
        processed = dataset.map(
            self.process_single_example,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            desc="Processing preference data",
            cache_file_name=cache_dir
        )
        
        return processed
    
    def validate_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Validate processed dataset and return statistics.
        
        Args:
            dataset: Processed dataset
        
        Returns:
            Validation statistics
        """
        
        if len(dataset) == 0:
            return {"error": "Dataset is empty"}
        
        # Check required columns
        required_cols = ["chosen_input_ids", "chosen_attention_mask", 
                        "rejected_input_ids", "rejected_attention_mask"]
        missing_cols = [col for col in required_cols if col not in dataset.column_names]
        
        if missing_cols:
            return {"error": f"Missing columns: {missing_cols}"}
        
        # Sample a few examples for validation
        sample = dataset[0]
        
        # Check tensor shapes
        stats = {
            "num_examples": len(dataset),
            "sequence_length": len(sample["chosen_input_ids"]),
            "vocab_size_check": {
                "max_chosen_token": int(torch.max(sample["chosen_input_ids"]).item()),
                "max_rejected_token": int(torch.max(sample["rejected_input_ids"]).item()),
                "tokenizer_vocab_size": self.tokenizer.vocab_size,
            },
            "padding_stats": {
                "chosen_pad_tokens": int(torch.sum(sample["chosen_input_ids"] == self.tokenizer.pad_token_id).item()),
                "rejected_pad_tokens": int(torch.sum(sample["rejected_input_ids"] == self.tokenizer.pad_token_id).item()),
            },
            "validation": "passed"
        }
        
        return stats


class EvaluationDataProcessor:
    """Processor for evaluation datasets used in GenSteer assessment."""
    
    def __init__(self, tokenizer: AutoTokenizer, max_input_length: int = 512):
        """
        Initialize evaluation data processor.
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_input_length: Maximum input sequence length
        """
        
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_generation_dataset(
        self,
        prompts: List[str],
        instructions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Prepare dataset for text generation evaluation.
        
        Args:
            prompts: List of input prompts
            instructions: Optional system instructions
        
        Returns:
            List of prepared examples
        """
        
        examples = []
        
        for i, prompt in enumerate(prompts):
            # Build full input
            if instructions and i < len(instructions):
                full_input = f"{instructions[i]}\n\n{prompt}"
            else:
                full_input = prompt
            
            # Tokenize
            tokens = self.tokenizer(
                full_input,
                max_length=self.max_input_length,
                truncation=True,
                return_tensors="pt"
            )
            
            examples.append({
                "prompt": prompt,
                "full_input": full_input,
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
                "instruction": instructions[i] if instructions and i < len(instructions) else None
            })
        
        return examples
    
    def load_standard_eval_dataset(
        self,
        dataset_name: str,
        split: str = "test",
        num_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load and process standard evaluation datasets.
        
        Args:
            dataset_name: Name of the dataset (e.g., "Anthropic/hh-rlhf")
            split: Dataset split to use
            num_samples: Maximum number of samples to load
        
        Returns:
            Processed evaluation examples
        """
        
        # Load dataset
        dataset = load_dataset(dataset_name, split=split)
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        examples = []
        
        for item in tqdm(dataset, desc=f"Processing {dataset_name}"):
            # Extract prompt (usually from chosen response)
            if "chosen" in item:
                # For preference datasets, extract the prompt part
                chosen_text = item["chosen"]
                
                # Simple heuristic to extract prompt (before assistant's response)
                if "Human:" in chosen_text and "Assistant:" in chosen_text:
                    human_parts = chosen_text.split("Human:")[-1].split("Assistant:")[0].strip()
                    prompt = human_parts
                else:
                    prompt = chosen_text[:200] + "..." if len(chosen_text) > 200 else chosen_text
                
            elif "prompt" in item:
                prompt = item["prompt"]
            else:
                # Skip if no recognizable prompt format
                continue
            
            # Tokenize
            tokens = self.tokenizer(
                prompt,
                max_length=self.max_input_length,
                truncation=True,
                return_tensors="pt"
            )
            
            examples.append({
                "prompt": prompt,
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
                "dataset": dataset_name,
                "original_item": item
            })
        
        return examples


class DataCollator:
    """Custom data collator for GenSteer training."""
    
    def __init__(self, tokenizer: AutoTokenizer, padding: str = "longest"):
        """
        Initialize data collator.
        
        Args:
            tokenizer: HuggingFace tokenizer
            padding: Padding strategy
        """
        
        self.tokenizer = tokenizer
        self.padding = padding
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of examples.
        
        Args:
            batch: List of processed examples
        
        Returns:
            Batched tensors
        """
        
        # Extract tensor keys
        tensor_keys = ["chosen_input_ids", "chosen_attention_mask", 
                      "rejected_input_ids", "rejected_attention_mask"]
        
        batched = {}
        
        for key in tensor_keys:
            if key in batch[0]:
                tensors = [item[key] for item in batch]
                
                if self.padding == "longest":
                    # Pad to longest in batch
                    max_len = max(tensor.size(0) for tensor in tensors)
                    padded_tensors = []
                    
                    for tensor in tensors:
                        if tensor.size(0) < max_len:
                            pad_length = max_len - tensor.size(0)
                            if "attention_mask" in key:
                                # Pad attention masks with 0
                                padded = torch.cat([tensor, torch.zeros(pad_length, dtype=tensor.dtype)])
                            else:
                                # Pad input_ids with pad_token_id
                                padded = torch.cat([tensor, torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=tensor.dtype)])
                            padded_tensors.append(padded)
                        else:
                            padded_tensors.append(tensor)
                    
                    batched[key] = torch.stack(padded_tensors)
                else:
                    # Default stacking (assumes same length)
                    batched[key] = torch.stack(tensors)
        
        return batched


def load_preference_dataset(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    split: str = "train",
    max_length: int = 1024,
    num_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    num_proc: int = 4
) -> Dataset:
    """
    Convenience function to load and process preference datasets.
    
    Args:
        dataset_name: Name of preference dataset
        tokenizer: HuggingFace tokenizer
        split: Dataset split
        max_length: Maximum sequence length
        num_samples: Maximum number of samples
        cache_dir: Cache directory
        num_proc: Number of processes
    
    Returns:
        Processed preference dataset
    """
    
    # Load raw dataset
    print(f"📚 Loading {dataset_name} ({split} split)...")
    dataset = load_dataset(dataset_name, split=split)
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        print(f"🔢 Selected {len(dataset)} samples")
    
    # Process dataset
    processor = PreferenceDataProcessor(
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    processed_dataset = processor.process_dataset(
        dataset=dataset,
        num_proc=num_proc,
        cache_dir=cache_dir
    )
    
    # Validate
    stats = processor.validate_dataset(processed_dataset)
    print(f"✅ Dataset processed: {stats['num_examples']} examples, max_length={stats['sequence_length']}")
    
    if stats.get("validation") != "passed":
        print(f"⚠️  Validation issues: {stats}")
    
    return processed_dataset


def create_evaluation_prompts(
    domain: str = "helpful_assistant",
    num_prompts: int = 50
) -> List[str]:
    """
    Create evaluation prompts for different domains.
    
    Args:
        domain: Evaluation domain
        num_prompts: Number of prompts to generate
    
    Returns:
        List of evaluation prompts
    """
    
    if domain == "helpful_assistant":
        base_prompts = [
            "How can I improve my productivity at work?",
            "What are some healthy meal prep ideas?",
            "Can you explain quantum computing in simple terms?",
            "What should I consider when buying a house?",
            "How do I start learning a new programming language?",
            "What are effective ways to manage stress?",
            "Can you recommend books for personal development?",
            "How do I create a budget and stick to it?",
            "What are the benefits of regular exercise?",
            "How can I improve my communication skills?",
        ]
    
    elif domain == "creative_writing":
        base_prompts = [
            "Write a short story about time travel.",
            "Create a poem about the changing seasons.",
            "Describe a futuristic city in vivid detail.",
            "Write a dialogue between two unlikely characters.",
            "Create a compelling character backstory.",
            "Write a mystery story with an unexpected twist.",
            "Describe a magical forest from a child's perspective.",
            "Create a haiku series about technology.",
            "Write an alternative ending to a famous story.",
            "Describe a day in the life of an AI assistant.",
        ]
    
    elif domain == "problem_solving":
        base_prompts = [
            "How would you solve traffic congestion in cities?",
            "What's the best approach to learning a difficult subject?",
            "How can we reduce food waste in households?",
            "What strategies help overcome procrastination?",
            "How do you resolve conflicts between team members?",
            "What's an effective way to organize digital files?",
            "How can small businesses compete with larger companies?",
            "What approach works best for breaking bad habits?",
            "How do you prioritize tasks when everything seems urgent?",
            "What methods help improve memory and learning retention?",
        ]
    
    else:
        # Generic prompts
        base_prompts = [
            "Tell me something interesting.",
            "What's your opinion on current technology trends?",
            "How has the world changed in recent years?",
            "What advice would you give to someone starting their career?",
            "Explain a complex topic in simple terms.",
        ]
    
    # Extend prompts if needed
    prompts = base_prompts * (num_prompts // len(base_prompts) + 1)
    return prompts[:num_prompts]


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("argsearch/llama-7b-sft-float32")
    
    # Test preference data processing
    print("🧪 Testing preference data processing...")
    
    # Load small sample
    dataset = load_preference_dataset(
        dataset_name="Anthropic/hh-rlhf",
        tokenizer=tokenizer,
        split="train",
        max_length=512,
        num_samples=100
    )
    
    print(f"✅ Processed dataset: {len(dataset)} examples")
    print(f"   Example keys: {dataset[0].keys()}")
    print(f"   Sequence length: {len(dataset[0]['chosen_input_ids'])}")
    
    # Test evaluation data processing
    print("\n🧪 Testing evaluation data processing...")
    
    eval_processor = EvaluationDataProcessor(tokenizer)
    prompts = create_evaluation_prompts("helpful_assistant", 10)
    eval_examples = eval_processor.prepare_generation_dataset(prompts)
    
    print(f"✅ Created {len(eval_examples)} evaluation examples")
    print(f"   Example prompt: {eval_examples[0]['prompt']}")
    print(f"   Input length: {len(eval_examples[0]['input_ids'])}")