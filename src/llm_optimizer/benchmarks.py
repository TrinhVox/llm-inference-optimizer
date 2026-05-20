import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from gptqmodel import GPTQModel, BACKEND
from awq import AutoAWQForCausalLM
from utils import MODEL_NAME, measure_inference


SEQUENCE_LENGTHS = [256, 512, 1024, 2048]
PROMPT = "Explain the concept of gravity in simple terms."

def load_fp16():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
    )
    return model, tokenizer

def load_bnb():
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config = quantization_config,
        device_map="auto"
    )
    return model, tokenizer

def load_gptq():
    quant_path = "Qwen2.5-0.5B-gptqmodel-4bit"
    model = GPTQModel.load(quant_path, backend=BACKEND.TRITON)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

def load_awq():
    model = AutoAWQForCausalLM.from_quantized("RichardErkhov/Qwen_-_Qwen2.5-0.5B-awq")
    tokenizer = AutoTokenizer.from_pretrained("RichardErkhov/Qwen_-_Qwen2.5-0.5B-awq")
    return model.model, tokenizer

MODELS = {
    "fp16": load_fp16,
    "bnb_4bit": load_bnb,
    "gptq_4bit": load_gptq,
    "awq_4bit": load_awq,
}

def run_benchmarks():
    all_results = {}

    for name, load_fn in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Benchmarking: {name}")
        print(f"{'='*50}")

        model, tokenizer = load_fn()
        all_results[name] = {}

        for seq_len in SEQUENCE_LENGTHS:
            print(f"\n  Sequence length: {seq_len}")
            result = measure_inference(model, tokenizer, PROMPT, max_new_tokens=seq_len)
            all_results[name][seq_len] = result

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    output_path = Path("../../results/benchmarks.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "results": all_results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    run_benchmarks()