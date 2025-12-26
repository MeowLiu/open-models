#!/usr/bin/env python3
"""调试模型加载过程"""

import os
import sys
import traceback
import logging

# 配置详细日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 禁用 Hugging Face Hub 的符号链接，避免 Windows 权限问题
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def debug_model_load():
    """调试模型加载"""
    from config import ModelsPath

    model_path = ModelsPath.ChenkinNoob.value
    logger.info(f"模型路径: {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        return False

    logger.info(f"路径存在，是目录: {os.path.isdir(model_path)}")

    if os.path.isdir(model_path):
        import glob
        safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        logger.info(f".safetensors文件数量: {len(safetensors_files)}")
        if safetensors_files:
            file_size = os.path.getsize(safetensors_files[0]) / (1024**3)  # GB
            logger.info(f"模型文件大小: {file_size:.2f} GB")

    print("\n=== 尝试导入模块 ===")
    try:
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
        logger.info("✓ 成功导入StableDiffusionXLPipeline")
    except ImportError as e:
        logger.error(f"导入失败: {e}")
        return False

    print("\n=== 检查CUDA状态 ===")
    import torch
    if torch.cuda.is_available():
        logger.info(f"CUDA可用，设备: {torch.cuda.get_device_name(0)}")
        logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.warning("CUDA不可用，将使用CPU")

    print("\n=== 尝试加载模型（单步调试）===")

    # 步骤1: 尝试使用from_single_file加载
    logger.info("步骤1: 尝试使用from_single_file加载...")
    try:
        import glob
        safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        if safetensors_files:
            model_file = safetensors_files[0]
            logger.info(f"使用模型文件: {model_file}")

            logger.info("调用StableDiffusionXLPipeline.from_single_file...")
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_file,
                torch_dtype=torch.float16,
            )
            logger.info("✓ from_single_file成功")

            # 测试优化
            logger.info("测试启用注意力切片...")
            pipe.enable_attention_slicing()
            logger.info("✓ 注意力切片已启用")

            # 检查是否需要CPU卸载
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"GPU显存: {total_memory:.1f} GB")

                if total_memory < 10:
                    logger.info("显存有限，启用CPU卸载...")
                    try:
                        pipe.enable_model_cpu_offload()
                        logger.info("✓ CPU卸载已启用")
                    except Exception as e:
                        logger.error(f"CPU卸载失败: {e}")
                else:
                    logger.info("尝试加载到GPU...")
                    try:
                        pipe = pipe.to("cuda")
                        logger.info("✓ 成功加载到GPU")
                    except Exception as e:
                        logger.error(f"加载到GPU失败: {e}")
                        logger.info("回退到CPU卸载...")
                        pipe.enable_model_cpu_offload()
                        logger.info("✓ 回退到CPU卸载")

            return True
        else:
            logger.error("未找到.safetensors文件")
            return False

    except Exception as e:
        logger.error(f"加载失败: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== 调试模型加载过程 ===")
    success = debug_model_load()
    if success:
        print("\n✓ 调试成功，模型可以加载")
    else:
        print("\n✗ 调试失败，模型加载有问题")
    sys.exit(0 if success else 1)