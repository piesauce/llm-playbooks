import numpy as np
import torch

# Define the numbers
numbers = np.array([2.3888888, 0, 34.444, 12.3486e4, -1223.4566], dtype=np.float32)

# Convert numbers to different quantized formats
numbers_f16 = numbers.astype(np.float16)
numbers_bf16 = torch.tensor(numbers).to(torch.bfloat16).numpy()
numbers_int8 = np.clip(numbers, -128, 127).astype(np.int8)  # int8 has range [-128, 127]

# Perform arithmetic operations (sum, product, mean)
def compute_operations(arr, dtype_name):
    print(f"Results for {dtype_name}:")
    print(f"  Sum: {arr.sum()}")
    print(f"  Product: {arr.prod()}")
    print(f"  Mean: {arr.mean()}")
    print()

# Compute operations for each data type
compute_operations(numbers, "float32")
compute_operations(numbers_f16, "float16")
compute_operations(numbers_bf16, "bfloat16")
compute_operations(numbers_int8, "int8")

# Compute precision loss
def compute_precision_loss(original, quantized, dtype_name):
    loss = np.abs(original - quantized)
    print(f"Precision loss for {dtype_name}:")
    print(loss)
    print()

compute_precision_loss(numbers, numbers_f16, "float16")
compute_precision_loss(numbers, numbers_bf16, "bfloat16")
compute_precision_loss(numbers, numbers_int8, "int8")
