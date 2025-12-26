#!/usr/bin/env python3
"""测试模型加载"""

import os
import sys
import traceback

# 禁用 Hugging Face Hub 的符号链接，避免 Windows 权限问题
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def test_model_load():
    """测试模型加载"""
    from config import ModelsPath

    model_path = ModelsPath.ChenkinNoob.value
    print(f"模型路径: {model_path}")
    print(f"路径是否存在: {os.path.exists(model_path)}")
    print(f"是否是文件: {os.path.isfile(model_path)}")
    print(f"是否是目录: {os.path.isdir(model_path)}")

    if os.path.isdir(model_path):
        import glob
        safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        print(f".safetensors文件数量: {len(safetensors_files)}")
        if safetensors_files:
            print(f"第一个文件: {safetensors_files[0]}")
            file_size = os.path.getsize(safetensors_files[0]) / (1024**3)  # GB
            print(f"文件大小: {file_size:.2f} GB")

    print("\n尝试加载模型...")
    try:
        from image_generator import ChenkinNoobImageGenerator
        print("导入成功")

        # 尝试创建实例
        print("正在创建ChenkinNoobImageGenerator实例...")
        generator = ChenkinNoobImageGenerator(model_path)
        print("✓ 模型加载成功!")

        # 测试生成小图像
        print("\n测试生成小图像...")
        import torch
        if torch.cuda.is_available():
            print(f"CUDA可用，设备: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("CUDA不可用，使用CPU")

    except Exception as e:
        print(f"✗ 错误: {e}")
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = test_model_load()
    sys.exit(0 if success else 1)