import torch
import os

# 禁用 Hugging Face Hub 的符号链接，避免 Windows 权限问题
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
import matplotlib.pyplot as plt
from config import ModelsPath

# 构建模型路径
model_path = ModelsPath.ChenkinNoob.value
# 检查路径是否存在
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型路径不存在: {model_path}")

# 如果是文件且为 .safetensors 格式，直接加载
if os.path.isfile(model_path) and model_path.endswith(".safetensors"):
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
    )
elif os.path.isdir(model_path):
    # 如果是文件夹，检查内部是否有 .safetensors 文件
    import glob
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    if safetensors_files:
        # 找到 .safetensors 文件，使用单文件加载
        pipe = StableDiffusionXLPipeline.from_single_file(
            safetensors_files[0],
            torch_dtype=torch.float16,
            local_files_only=True,
        )
    else:
        # 没有 .safetensors 文件，假定为 Diffusers 格式
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True,
        )
else:
    raise ValueError(f"模型路径既不是文件也不是文件夹: {model_path}")

# 应用显存优化
print("应用显存优化...")

# 启用注意力切片（减少峰值显存使用）
pipe.enable_attention_slicing()

# 启用VAE tiled解码（减少大图像解码时的显存使用）
try:
    pipe.vae.enable_tiling()
    print("已启用VAE tiled解码")
except Exception:
    print("注意：当前VAE不支持tiled解码")

# 如果有GPU，将模型移到GPU上，并考虑CPU卸载
if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    print(f"GPU显存: {total_memory:.1f} GB")

    # 根据显存大小选择优化策略
    if total_memory < 8:  # 小于8GB显存，使用CPU卸载
        print("显存较小，启用CPU卸载...")
        pipe.enable_model_cpu_offload()
    else:
        # 显存足够，直接加载到GPU
        pipe = pipe.to("cuda")
        print("已使用CUDA加速")
else:
    print("未检测到CUDA GPU，使用CPU模式")

# 生成图像
prompt = "cat girl"  # 使用英文提示词效果更好
negative_prompt = "low quality, worst quality, blurry, deformed"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,  # 推理步数，建议20-50
    guidance_scale=7.5,  # 指导比例，建议5-10
    height=1024,  # 图像高度
    width=1024,  # 图像宽度
    generator=torch.manual_seed(42),  # 随机种子，可固定结果
).images[0]

# 保存和显示图像
image.save("generated_images/generated_image.png")
plt.imshow(image)
plt.axis("off")
plt.show()
