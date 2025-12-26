#!/usr/bin/env python3
"""只测试PyTorch和CUDA，不加载模型"""

import torch
import sys

print("=== PyTorch/CUDA 纯测试 ===")

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"设备数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  设备 {i}: {props.name}")
        print(f"    显存: {props.total_memory / 1024**3:.1f} GB")

    print("\n=== 简单CUDA计算 ===")
    try:
        # 最基本的CUDA张量操作
        a = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        b = torch.tensor([4.0, 5.0, 6.0], device='cuda')
        c = a + b
        print(f"计算成功: {a.cpu().numpy()} + {b.cpu().numpy()} = {c.cpu().numpy()}")

        # 测试显存分配（小量）
        small = torch.randn(1000, 1000, device='cuda')  # 约4MB
        print(f"小显存分配成功: {small.shape}")

        # 测试大显存分配（100MB）
        large = torch.randn(5000, 5000, device='cuda')  # 约100MB
        print(f"大显存分配成功: {large.shape}")

        del small, large
        torch.cuda.empty_cache()

        print("\n✓ CUDA测试全部通过")

    except Exception as e:
        print(f"✗ CUDA测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n=== 测试完成 ===")