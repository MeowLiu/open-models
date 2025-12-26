#!/usr/bin/env python3
"""最小化测试：尝试加载模型并生成512x512图像"""

import os
import sys
import traceback
import time

# 设置环境变量
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def test_minimal():
    """最小化测试"""
    print("=== 最小化测试: 512x512图像生成 ===")

    from config import ModelsPath
    model_path = ModelsPath.ChenkinNoob.value
    print(f"模型路径: {model_path}")

    print("\n1. 检查文件...")
    if not os.path.exists(model_path):
        print("错误: 模型路径不存在")
        return False

    print("2. 导入模块...")
    try:
        from image_generator import ChenkinNoobImageGenerator
        print("导入成功")
    except ImportError as e:
        print(f"导入失败: {e}")
        return False

    print("3. 创建生成器实例...")
    try:
        start_time = time.time()
        generator = ChenkinNoobImageGenerator(model_path)
        load_time = time.time() - start_time
        print(f"✓ 生成器创建成功 ({load_time:.1f}秒)")
    except Exception as e:
        print(f"✗ 创建生成器失败: {e}")
        traceback.print_exc()
        return False

    print("4. 生成512x512测试图像...")
    try:
        prompt = "simple test"
        negative_prompt = ""

        start_time = time.time()
        output_path = generator.generate_and_save(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=512,
            height=512,
            output_path="test_output_512.png",
            num_inference_steps=4,  # 最小步数用于测试
            guidance_scale=7.5,
            seed=42,
            show_preview=False,
        )
        gen_time = time.time() - start_time

        print(f"✓ 图像生成成功 ({gen_time:.1f}秒)")
        print(f"输出路径: {output_path}")

        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024  # KB
            print(f"文件大小: {file_size:.1f} KB")
            # 清理测试文件
            os.remove(output_path)
            print("测试文件已清理")

        return True

    except Exception as e:
        print(f"✗ 图像生成失败: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试...")
    success = test_minimal()
    if success:
        print("\n✓ 所有测试通过!")
        sys.exit(0)
    else:
        print("\n✗ 测试失败")
        sys.exit(1)