import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
from utils import MODEL_NAME, PROMPTS, measure_inference




def load_model_4bit():
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config = quantization_config,
        device_map="auto"
    )
    return model, tokenizer






def run_4bit():
    print("Loading model in with quantization")
    model, tokenizer = load_model_4bit()
    
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
    output_path = Path("../../results/bnb_4bit.json")
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
    run_4bit()