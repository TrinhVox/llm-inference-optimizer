import torch
import json
from pathlib import Path 
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from gptqmodel import GPTQModel, AWQConfig, BACKEND
from utils import MODEL_NAME, PROMPTS, measure_inference
from awq import AutoAWQForCausalLM
import argparse





def load_model_awq():
    model = AutoAWQForCausalLM.from_quantized("RichardErkhov/Qwen_-_Qwen2.5-0.5B-awq")
    tokenizer = AutoTokenizer.from_pretrained("RichardErkhov/Qwen_-_Qwen2.5-0.5B-awq")
    return model.model, tokenizer




def run_awq():
    print("Loading model in with quantization")
    
    model, tokenizer = load_model_awq()
    model_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"Model loaded. GPU memory: {model_mem:.0f} MB")
    
    results = []
    for i, prompt in enumerate(PROMPTS):
        print(f"\nRunning prompt {i+1}/{len(PROMPTS)}...")
        result = measure_inference(model, tokenizer, prompt)
        results.append(result)
        print(f"  Tokens/sec: {result['tokens_per_sec']}")
        print(f"  Peak memory: {result['peak_memory_mb']} MB")
    
    # Save results
    output_path = Path("../../results/awq.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "dtype": "auto",
            "model_memory_mb": round(model_mem, 2),
            "results": results,
        }, f, indent=2)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    run_awq()