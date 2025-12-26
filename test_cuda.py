#!/usr/bin/env python3
"""测试CUDA和PyTorch基本功能"""

import torch
import sys

print("=== PyTorch/CUDA 基本测试 ===")

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

    print("\n=== CUDA简单计算测试 ===")
    try:
        # 创建一个小张量在GPU上
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        y = torch.tensor([4.0, 5.0, 6.0]).cuda()
        z = x + y
        print(f"CUDA计算测试: {x} + {y} = {z}")
        print("✓ CUDA基本计算正常")

        # 测试显存分配
        print("\n=== 显存分配测试 ===")
        # 尝试分配500MB显存
        size_mb = 500
        size = size_mb * 1024 * 1024 // 4  # 浮点数数量 (4字节每个)
        try:
            big_tensor = torch.randn(size, device='cuda')
            print(f"✓ 成功分配{size_mb}MB显存")
            del big_tensor
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"✗ 分配{size_mb}MB显存失败: {e}")

    except Exception as e:
        print(f"✗ CUDA测试失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print("警告: CUDA不可用")

print("\n=== CPU内存测试 ===")
import psutil
mem = psutil.virtual_memory()
print(f"总内存: {mem.total / 1024**3:.1f} GB")
print(f"可用内存: {mem.available / 1024**3:.1f} GB")
print(f"内存使用率: {mem.percent}%")

# 检查是否有足够内存加载模型
model_size_gb = 6.62
required_memory_gb = model_size_gb * 2  # 保守估计需要2倍模型大小的内存

if mem.available / 1024**3 < required_memory_gb:
    print(f"\n⚠ 警告: 可用内存({mem.available / 1024**3:.1f}GB)可能不足")
    print(f"  模型大小: {model_size_gb:.1f}GB")
    print(f"  建议可用内存: >{required_memory_gb:.1f}GB")
    print("  建议: 关闭其他应用程序释放内存")

print("\n=== 测试完成 ===")