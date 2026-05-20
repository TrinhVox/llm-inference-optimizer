import torch
import json
from pathlib import Path 
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from gptqmodel import GPTQModel, GPTQConfig, BACKEND
from utils import MODEL_NAME, PROMPTS, measure_inference
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--quantize", action="store_true", help="Run quantization, otherwise load saved model")
args = parser.parse_args()


def quantize_model_gptq():
    calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
    ).select(range(1024))["text"]
    quant_path = "Qwen2.5-0.5B-gptqmodel-4bit"
    quant_config = GPTQConfig(bits=4, group_size=128)
    model = GPTQModel.load(MODEL_NAME, quant_config)
    model.quantize(calibration_dataset, batch_size = 4)
    model.save(quant_path)
    gptq_model = GPTQModel.load(quant_path, backend = BACKEND.TRITON)
    return gptq_model




def run_gptq():
    print("Loading model in with quantization")
    if args.quantize:
        model = quantize_model_gptq()
    else:
        quant_path = "Qwen2.5-0.5B-gptqmodel-4bit"
        model = GPTQModel.load(quant_path, backend = BACKEND.TRITON)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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
    output_path = Path("../../results/gptq.json")
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
    run_gptq()