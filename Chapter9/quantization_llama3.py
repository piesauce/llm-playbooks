import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name, dtype):
    """Load Llama model with specified dtype."""
    return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

def evaluate_model(model, tokenizer, prompt):
    """Run inference and measure time."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50)
    end_time = time.time()
    
    return end_time - start_time

def get_model_size(model):
    """Calculate model size in MB."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1"  # Change to actual model path
    dataset_sample_prompt = "Once upon a time, in a land far away,"  # Example prompt
    
    dtype_options = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "int8": "int8"  # For int8 quantization
    }
    
    results = {}
    
    for dtype_name, dtype in dtype_options.items():
        print(f"Evaluating {dtype_name} precision...")
        
        if dtype_name == "int8":
            model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
        else:
            model = load_model(model_name, dtype)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        inference_time = evaluate_model(model, tokenizer, dataset_sample_prompt)
        model_size = get_model_size(model) if dtype_name != "int8" else "Reduced (8-bit)"
        
        results[dtype_name] = {
            "Inference Time (s)": inference_time,
            "Model Size (MB)": model_size
        }
        
    print("\nResults:")
    for dtype, metrics in results.items():
        print(f"{dtype}: {metrics}")
