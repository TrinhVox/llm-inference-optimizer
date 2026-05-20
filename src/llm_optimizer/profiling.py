import torch
import json
from torch.profiler import profile, ProfilerActivity
from pathlib import Path
from utils import MODEL_NAME, measure_inference
from benchmarks import load_fp16, load_bnb, load_gptq, load_awq

PROMPT = "Explain the concept of gravity in simple terms."

def profile_model(name, model, tokenizer):
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    
    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1)
    
    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True
    ) as prof:
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=100, eos_token_id=None)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace(f"../../results/trace_{name}.json")

MODELS = {
    "fp16": load_fp16,
    "bnb_4bit": load_bnb,
    "gptq_4bit": load_gptq,
    "awq_4bit": load_awq,
}

def run_profiling():
    for name, load_fn in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Profiling: {name}")
        print(f"{'='*50}")

        model, tokenizer = load_fn()

        profile_model(name, model, tokenizer)

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_profiling()