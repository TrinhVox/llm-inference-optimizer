import torch
import time
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

PROMPTS = [
    "Explain the concept of gravity in simple terms.",
    "Write a short story about a robot learning to paint.",
    "What are the main differences between Python and C++?",
    "Summarize the history of the internet in one paragraph.",
    "A patient presents with chest pain and shortness of breath. What are the possible diagnoses?",
]



def load_model_fp16():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
    )
    return model, tokenizer


def measure_inference(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1)
    
    torch.cuda.synchronize() # Wait all GPU operations to finish
    torch.cuda.reset_peak_memory_stats() # Reset memory tracker
    
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens) # input prompt + generated tokens
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    total_time = end - start
    generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    tokens_per_sec = generated_tokens / total_time
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        "prompt": prompt,
        "generated_tokens": generated_tokens,
        "total_time_s": round(total_time, 4),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "peak_memory_mb": round(peak_memory_mb, 2),
        "output_preview": decoded[:200],
    }

def run_baseline():
    print("Loading model in FP16...")
    model, tokenizer = load_model_fp16()
    
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
    output_path = Path("results/baseline_fp16.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "dtype": "float16",
            "model_memory_mb": round(model_mem, 2),
            "results": results,
        }, f, indent=2)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    run_baseline()