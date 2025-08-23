#!/usr/bin/env python3
"""
GenSteer Response Generation Script
Paper: GenSteer: A Generative Steering Engine for Autonomous Test-Time Alignment

Command-line interface for generating responses with trained GenSteer models.
"""

import argparse
import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
import time

from inference import load_gensteer_inference
from data import create_evaluation_prompts


def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompts from various file formats."""
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    
    if file_path.suffix == ".json":
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return [str(item) for item in data]
            elif isinstance(data, dict) and "prompts" in data:
                return [str(item) for item in data["prompts"]]
            else:
                raise ValueError("JSON file must contain a list of prompts or a dict with 'prompts' key")
    
    elif file_path.suffix in [".txt", ".md"]:
        with open(file_path, 'r') as f:
            content = f.read().strip()
            # Split by double newlines to separate prompts
            prompts = [prompt.strip() for prompt in content.split('\n\n') if prompt.strip()]
            return prompts
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def save_results(results: List[Dict[str, Any]], output_file: str, format_type: str = "json"):
    """Save generation results to file."""
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format_type == "json":
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    elif format_type == "jsonl":
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    elif format_type == "txt":
        with open(output_path, 'w') as f:
            for i, result in enumerate(results, 1):
                f.write(f"=== Prompt {i} ===\n")
                f.write(f"{result['prompt']}\n\n")
                f.write(f"=== GenSteer Response ===\n")
                f.write(f"{result.get('gensteer_response', result.get('response', ''))}\n\n")
                
                if 'steering_info' in result:
                    steering_info = result['steering_info']
                    f.write(f"=== Steering Info ===\n")
                    f.write(f"Average Steering: {steering_info.get('avg_steering_strength', 0):.3f}\n")
                    f.write(f"Steering Std: {steering_info.get('std_steering_strength', 0):.3f}\n")
                    if 'learned_steering_strengths' in steering_info:
                        strengths = steering_info['learned_steering_strengths']
                        f.write(f"Steering Range: {min(strengths):.3f} - {max(strengths):.3f}\n")
                
                if 'base_response' in result:
                    f.write(f"\n=== Base Model Response ===\n")
                    f.write(f"{result['base_response']}\n")
                
                f.write("\n" + "="*60 + "\n\n")
    
    else:
        raise ValueError(f"Unsupported output format: {format_type}")
    
    print(f"💾 Results saved to {output_path}")


def interactive_mode(inference_engine):
    """Interactive chat mode."""
    
    print("🤖 GenSteer Interactive Mode")
    print("Type 'quit', 'exit', or press Ctrl+C to stop")
    print("Type 'compare' to compare with base model")
    print("Type 'info' to show model information")
    print("-" * 50)
    
    compare_mode = False
    
    try:
        while True:
            # Get user input
            try:
                user_input = input("\n👤 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit']:
                break
            elif user_input.lower() == 'compare':
                compare_mode = not compare_mode
                status = "ON" if compare_mode else "OFF"
                print(f"🔄 Base model comparison: {status}")
                continue
            elif user_input.lower() == 'info':
                info = inference_engine.get_model_info()
                print(f"📊 Model: {info['model_type']}")
                print(f"📁 Checkpoint: {Path(info['checkpoint_path']).name}")
                if 'training_info' in info and 'metrics' in info['training_info']:
                    metrics = info['training_info']['metrics']
                    if 'avg_steering_strength' in metrics:
                        print(f"🔧 Training Steering: {metrics['avg_steering_strength']:.3f}±{metrics.get('std_steering_strength', 0):.3f}")
                continue
            
            # Generate response
            print("🤖 GenSteer:", end=" ", flush=True)
            start_time = time.time()
            
            if compare_mode:
                result = inference_engine.compare_with_base(user_input, max_length=200)
                print(result['gensteer_response'])
                
                steering_info = result['steering_info']
                print(f"\n🔧 Steering: {steering_info.get('avg_steering_strength', 0):.3f} (Active: {result['steering_active']})")
                print(f"📝 Base Model: {result['base_response']}")
            
            else:
                response, steering_info = inference_engine.generate_response(
                    user_input, 
                    max_length=200, 
                    return_steering_info=True
                )
                print(response)
                
                avg_steering = steering_info.get('avg_steering_strength', 0)
                if avg_steering > 0.1:
                    print(f"\n🔧 Steering: {avg_steering:.3f}")
            
            elapsed = time.time() - start_time
            print(f"\n⏱️  Generated in {elapsed:.1f}s")
    
    except KeyboardInterrupt:
        pass
    
    print("\n👋 Goodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses using trained GenSteer models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prompt generation
  python generate.py --checkpoint model.pt --prompt "Hello, how are you?"
  
  # Batch generation from file
  python generate.py --checkpoint model.pt --prompts_file prompts.json --output results.json
  
  # Interactive mode
  python generate.py --checkpoint model.pt --interactive
  
  # Compare with base model
  python generate.py --checkpoint model.pt --prompt "Hello" --compare
  
  # Generate evaluation prompts
  python generate.py --checkpoint model.pt --eval_domain helpful_assistant --num_eval 20
        """
    )
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to GenSteer checkpoint")
    parser.add_argument("--base_model", type=str, default="argsearch/llama-7b-sft-float32", help="Base model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")
    
    # Input arguments
    parser.add_argument("--prompt", type=str, help="Single prompt to generate response for")
    parser.add_argument("--prompts_file", type=str, help="File containing prompts (JSON, JSONL, or TXT)")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    
    # Evaluation prompts
    parser.add_argument("--eval_domain", type=str, choices=["helpful_assistant", "creative_writing", "problem_solving", "generic"], help="Domain for evaluation prompts")
    parser.add_argument("--num_eval", type=int, default=10, help="Number of evaluation prompts to generate")
    
    # Generation parameters
    parser.add_argument("--max_length", type=int, default=256, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--no_sample", action="store_true", help="Use greedy decoding instead of sampling")
    
    # Output arguments
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--format", type=str, choices=["json", "jsonl", "txt"], default="json", help="Output format")
    parser.add_argument("--compare", action="store_true", help="Compare with base model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    
    # Analysis
    parser.add_argument("--analyze_steering", action="store_true", help="Analyze steering patterns")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.prompt, args.prompts_file, args.interactive, args.eval_domain]):
        parser.error("Must specify one of: --prompt, --prompts_file, --interactive, or --eval_domain")
    
    # Load inference engine
    print(f"🚀 Loading GenSteer from {args.checkpoint}")
    inference_engine = load_gensteer_inference(
        checkpoint_path=args.checkpoint,
        base_model_name=args.base_model,
        device=args.device
    )
    
    # Handle interactive mode
    if args.interactive:
        interactive_mode(inference_engine)
        return
    
    # Prepare prompts
    prompts = []
    
    if args.prompt:
        prompts = [args.prompt]
    
    elif args.prompts_file:
        prompts = load_prompts_from_file(args.prompts_file)
        print(f"📚 Loaded {len(prompts)} prompts from {args.prompts_file}")
    
    elif args.eval_domain:
        prompts = create_evaluation_prompts(args.eval_domain, args.num_eval)
        print(f"🎯 Created {len(prompts)} evaluation prompts for {args.eval_domain}")
    
    if not prompts:
        print("❌ No prompts to process")
        return
    
    # Generate responses
    print(f"⚡ Generating responses for {len(prompts)} prompts...")
    
    do_sample = not args.no_sample
    results = []
    
    if args.compare:
        # Compare mode
        for i, prompt in enumerate(prompts, 1):
            print(f"🔄 Processing {i}/{len(prompts)}: {prompt[:50]}...")
            
            result = inference_engine.compare_with_base(
                prompt=prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=do_sample
            )
            
            results.append(result)
    
    else:
        # Regular generation
        if len(prompts) > 5 and args.batch_size > 1:
            # Batch processing
            results = inference_engine.batch_generate(
                prompts=prompts,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=do_sample,
                batch_size=args.batch_size
            )
        else:
            # Individual processing with progress
            for i, prompt in enumerate(prompts, 1):
                print(f"⚡ Processing {i}/{len(prompts)}: {prompt[:50]}...")
                
                response, steering_info = inference_engine.generate_response(
                    prompt=prompt,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=do_sample,
                    return_steering_info=True
                )
                
                results.append({
                    "prompt": prompt,
                    "response": response,
                    "steering_info": steering_info
                })
    
    # Analyze steering patterns if requested
    if args.analyze_steering and not args.compare:
        print("\n🔍 Analyzing steering patterns...")
        steering_analysis = inference_engine.analyze_steering_patterns(prompts)
        
        if "error" not in steering_analysis:
            stats = steering_analysis["steering_statistics"]
            print(f"📊 Steering Analysis:")
            print(f"   Average: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"   Range: {stats['min']:.3f} - {stats['max']:.3f}")
            print(f"   Active Steering: {steering_analysis['utilization']['active_steering_ratio']*100:.1f}%")
            print(f"   High Steering: {steering_analysis['utilization']['high_steering_ratio']*100:.1f}%")
            
            # Add analysis to results
            for result in results:
                result["steering_analysis"] = steering_analysis
    
    # Output results
    if args.output:
        save_results(results, args.output, args.format)
    else:
        # Print to console
        for i, result in enumerate(results, 1):
            print(f"\n{'='*60}")
            print(f"Prompt {i}: {result['prompt']}")
            print(f"{'='*60}")
            
            if 'gensteer_response' in result:
                print(f"🤖 GenSteer: {result['gensteer_response']}")
                if result.get('steering_active', False):
                    steering_info = result['steering_info']
                    print(f"🔧 Steering: {steering_info.get('avg_steering_strength', 0):.3f}")
                
                if 'base_response' in result:
                    print(f"\n📝 Base: {result['base_response']}")
            
            else:
                print(f"🤖 Response: {result['response']}")
                steering_info = result['steering_info']
                avg_steering = steering_info.get('avg_steering_strength', 0)
                if avg_steering > 0.1:
                    print(f"🔧 Steering: {avg_steering:.3f}")
    
    print(f"\n✅ Generated {len(results)} responses")
    
    # Summary statistics
    if results and 'steering_info' in results[0]:
        all_steering = []
        for result in results:
            steering_info = result.get('steering_info', {})
            if 'avg_steering_strength' in steering_info:
                all_steering.append(steering_info['avg_steering_strength'])
        
        if all_steering:
            import numpy as np
            print(f"📊 Average steering across all responses: {np.mean(all_steering):.3f} ± {np.std(all_steering):.3f}")
            print(f"🎯 Steering utilization: {np.mean(np.array(all_steering) > 0.5)*100:.1f}%")


if __name__ == "__main__":
    main()