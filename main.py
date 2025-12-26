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
model_path = ModelsPath.ChenkinNoob
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
        )
    else:
        # 没有 .safetensors 文件，假定为 Diffusers 格式
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
else:
    raise ValueError(f"模型路径既不是文件也不是文件夹: {model_path}")

# 如果有GPU，将模型移到GPU上
if torch.cuda.is_available():
    pipe = pipe.to("cuda")
    print("已使用CUDA加速")

# 生成图像
prompt = "masterpiece, best quality, 1girl, beautiful detailed eyes, anime style"  # 使用英文提示词效果更好
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
