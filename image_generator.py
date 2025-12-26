import torch
import os
import glob
import sys
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
import matplotlib.pyplot as plt


class ChenkinNoobImageGenerator:
    """图像生成器类，支持加载.safetensors单文件或Diffusers目录格式的模型。"""

    def __init__(self, model_path: str):
        """
        初始化图像生成器。

        Args:
            model_path: 模型路径，可以是.safetensors文件或包含Diffusers格式模型的目录

        Raises:
            FileNotFoundError: 如果模型路径不存在
            ValueError: 如果模型路径既不是文件也不是文件夹，或者格式不支持
        """
        # 禁用 Hugging Face Hub 的符号链接，避免 Windows 权限问题
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        self.model_path = model_path
        self.pipe: StableDiffusionXLPipeline | None = None

        # 检查路径是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        self._load_model()
        self._apply_memory_optimizations()

    def _load_model(self):
        """加载模型，支持.safetensors单文件或Diffusers目录格式。"""
        # 如果是文件且为 .safetensors 格式，直接加载
        if os.path.isfile(self.model_path) and self.model_path.endswith(".safetensors"):
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16,
            )
        elif os.path.isdir(self.model_path):
            # 如果是文件夹，检查内部是否有 .safetensors 文件
            safetensors_files = glob.glob(
                os.path.join(self.model_path, "*.safetensors")
            )
            if safetensors_files:
                # 找到 .safetensors 文件，使用单文件加载
                self.pipe = StableDiffusionXLPipeline.from_single_file(
                    safetensors_files[0],
                    torch_dtype=torch.float16,
                    local_files_only=True,
                )
            else:
                # 没有 .safetensors 文件，假定为 Diffusers 格式
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    local_files_only=True,
                )
        else:
            raise ValueError(f"模型路径既不是文件也不是文件夹: {self.model_path}")

    def _apply_memory_optimizations(self):
        """应用显存优化策略。"""
        print("应用显存优化...")

        # 检查pipe是否成功加载
        if self.pipe is None:
            raise RuntimeError("模型管道未正确加载，无法应用优化")

        # 启用注意力切片（减少峰值显存使用）
        self.pipe.enable_attention_slicing()

        # 启用VAE tiled解码（减少大图像解码时的显存使用）
        try:
            self.pipe.vae.enable_tiling()
            print("已启用VAE tiled解码")
        except Exception:
            print("注意：当前VAE不支持tiled解码")

        # 如果有GPU，将模型移到GPU上，并考虑CPU卸载
        if torch.cuda.is_available():
            total_memory = (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
            )  # GB
            print(f"GPU显存: {total_memory:.1f} GB")

            # 对于大模型（>6GB），6GB显存肯定不够，必须使用CPU卸载
            # 保守阈值设为8GB，因为模型加载还需要额外内存
            if total_memory < 8:
                print(f"警告: 您的GPU显存({total_memory:.1f}GB)可能不足以加载此模型(6.6GB)。")
                print("将强制使用CPU卸载模式，生成速度会较慢。")
                print("建议: 如果可能，升级到至少8GB显存的GPU。")

            # 强制使用CPU卸载以确保稳定性
            print("启用CPU卸载以确保稳定性...")
            try:
                self.pipe.enable_model_cpu_offload()
                print("✓ CPU卸载已启用（模型在CPU，推理时按需加载到GPU）")
            except Exception as e:
                print(f"警告: CPU卸载失败: {e}")
                print("尝试直接加载到GPU（可能因显存不足而失败）...")
                try:
                    self.pipe = self.pipe.to("cuda")
                    print("已使用CUDA加速（高风险，可能崩溃）")
                except Exception as e2:
                    print(f"错误: 无法加载模型到GPU: {e2}")
                    raise RuntimeError("模型加载失败，显存不足。建议：\n" +
                                      "1. 关闭其他占用显存的应用程序\n" +
                                      "2. 使用更小的分辨率（如512x512）\n" +
                                      "3. 确保系统有至少12GB可用内存")
        else:
            print("未检测到CUDA GPU，使用CPU模式")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = 42,
    ) -> "PIL.Image.Image":
        """
        生成图像。

        Args:
            prompt: 正向提示词
            negative_prompt: 反向提示词
            width: 图像宽度（必须≥512且为64的倍数）
            height: 图像高度（必须≥512且为64的倍数）
            num_inference_steps: 推理步数
            guidance_scale: 指导比例
            seed: 随机种子

        Returns:
            生成的PIL图像对象

        Raises:
            ValueError: 如果宽度或高度不合法
        """
        # 验证尺寸
        if width < 512 or height < 512:
            raise ValueError(f"图像尺寸必须≥512x512，当前为{width}x{height}")
        if width % 64 != 0 or height % 64 != 0:
            raise ValueError(f"图像尺寸必须是64的倍数，当前为{width}x{height}")

        # 设置随机种子
        generator = torch.manual_seed(seed)

        # 检查管道是否加载成功
        if self.pipe is None:
            raise RuntimeError("模型管道未正确加载，请检查模型路径")

        # 生成图像
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        ).images[0]

        return image

    def generate_and_save(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        output_path: str = "generated_images/generated_image.png",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = 42,
        show_preview: bool = True,
    ) -> str:
        """
        生成图像并保存到文件。

        Args:
            prompt: 正向提示词
            negative_prompt: 反向提示词
            width: 图像宽度
            height: 图像高度
            output_path: 输出文件路径
            num_inference_steps: 推理步数
            guidance_scale: 指导比例
            seed: 随机种子
            show_preview: 是否显示预览

        Returns:
            保存的文件路径
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 生成图像
        image = self.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        # 保存图像
        image.save(output_path)
        print(f"图像已保存至: {output_path}")

        # 显示预览
        if show_preview:
            plt.imshow(image)
            plt.axis("off")
            plt.show()

        return output_path
