#!/usr/bin/env python3
"""简单测试CUDA和内存状态"""

import torch
import os
import sys

print("=== 系统信息 ===")
print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"设备数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  设备 {i}: {props.name}")
        print(f"    显存: {props.total_memory / 1024**3:.1f} GB")
        print(f"    计算能力: {props.major}.{props.minor}")
else:
    print("警告: CUDA不可用，将使用CPU模式")

print("\n=== 内存信息 ===")
import psutil
mem = psutil.virtual_memory()
print(f"总内存: {mem.total / 1024**3:.1f} GB")
print(f"可用内存: {mem.available / 1024**3:.1f} GB")
print(f"内存使用率: {mem.percent}%")

print("\n=== 模型文件检查 ===")
model_path = r"D:\open-models\models\ChenkinNoob\ChenkinNoob-XL-V0___2"
if os.path.exists(model_path):
    print(f"模型路径存在: {model_path}")
    if os.path.isdir(model_path):
        import glob
        safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        print(f".safetensors文件数量: {len(safetensors_files)}")
        if safetensors_files:
            file_size = os.path.getsize(safetensors_files[0]) / (1024**3)
            print(f"模型文件大小: {file_size:.2f} GB")
            print(f"需要内存估计: {file_size * 2:.1f}~{file_size * 3:.1f} GB (用于加载和推理)")
else:
    print(f"模型路径不存在: {model_path}")