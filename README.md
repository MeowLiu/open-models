# Open Models - AI 图像生成工具集

一个专门用于下载、管理和使用开源 AI 模型（特别是 ChenkinNoob 模型）的 Python 工具集。项目主要关注中文社区模型（如 ModelScope 平台）的下载和图像生成。

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

## ✨ 主要功能

- **🚀 交互式图像生成**：基于 ChenkinNoob 的 AI 图像生成，支持自定义提示词、尺寸和模型
- **📦 模型下载管理**：从 ModelScope 社区下载和管理中文模型，支持版本控制和本地缓存
- **🎨 智能显存优化**：自动根据可用显存启用注意力切片、VAE tiled 解码和 CPU 卸载
- **🖥️ 美观 CLI 界面**：类似 Vite 的交互式命令行界面，使用 InquirerPy 和 rich 库
- **📁 动态模型发现**：自动扫描 models 目录，支持 .safetensors 单文件和 Diffusers 目录格式

## 📋 系统要求

### 最低配置
- **操作系统**：Windows 10/11, Linux, macOS
- **Python**：3.10 或更高版本
- **内存**：12GB 系统内存
- **存储空间**：至少 15GB 可用空间（用于模型存储）
- **GPU**：可选，但推荐用于加速

### 推荐配置
- **GPU 显存**：8GB+（用于 768×768 分辨率）
- **系统内存**：16GB+
- **存储空间**：20GB+（用于多个模型）

### 最佳配置
- **GPU 显存**：12GB+（用于 1024×1024 分辨率）
- **系统内存**：32GB+
- **存储空间**：50GB+（用于大型模型库）

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/MeowLiu/open-models.git
cd open-models
```

### 2. 安装依赖
项目使用 [uv](https://github.com/astral-sh/uv) 作为包管理器（推荐）：
```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装项目依赖
uv sync
```

或者使用 pip：
```bash
pip install -e .
```

### 3. 下载模型
```bash
# 下载默认模型（ChenkinNoob-XL-V0.2，约 6.6GB）
python model_downloader.py ChenkinNoob/ChenkinNoob-XL-V0.2
```

### 4. 生成图像
```bash
# 交互式模式（推荐）
python main.py

# 或直接生成模式
python main.py --direct --prompt "1girl with cat ears" --width 1024 --height 1024
```

## 📁 项目结构

```
open-models/
├── cli.py                    # 交互式 CLI 主程序
├── main.py                   # 主入口点（支持交互式和直接模式）
├── image_generator.py        # 图像生成核心类
├── model_downloader.py       # 模型下载工具
├── config.py                 # 配置和模型路径管理
├── pyproject.toml            # 项目配置和依赖
├── uv.lock                   # 依赖锁定文件
├── README.md                 # 本文档
├── .gitignore               # Git 忽略规则
├── models/                  # 模型存储目录（自动创建）
├── generated_images/        # 生成的图像目录（自动创建）             
├── test_simple.py       # 简单系统测试
├── test_load.py         # 模型加载测试
├── test_cuda.py         # CUDA 功能测试
└── test_minimal.py      # 最小化图像生成测试
```

## 🔧 详细使用指南

### 模型下载工具

```bash
# 基本用法
python model_downloader.py [MODEL_ID] [OPTIONS]

# 示例：下载 ChenkinNoob 模型
python model_downloader.py ChenkinNoob/ChenkinNoob-XL-V0.2

# 列出本地可用的模型
python model_downloader.py --list

# 强制重新下载（即使已存在）
python model_downloader.py ChenkinNoob/ChenkinNoob-XL-V0.2 --force

# 使用特定版本和忽略模式
python model_downloader.py ChenkinNoob/ChenkinNoob-XL-V0.2 --revision v1.0 --ignore-pattern "*.msgpack"

# 自定义缓存目录
python model_downloader.py ChenkinNoob/ChenkinNoob-XL-V0.2 --cache-dir /path/to/models
```

**支持的模型格式**：
- 任何 ModelScope 格式的模型（格式：`username/model_name`）
- 默认推荐模型：`ChenkinNoob/ChenkinNoob-XL-V0.2`（Stable Diffusion XL 变体）

### 图像生成工具

#### 交互式 CLI 模式（默认）
```bash
python main.py
```
运行后将引导您完成：
1. **选择模型**：从本地可用模型中选择，或使用默认模型
2. **设置尺寸**：选择预设尺寸或自定义（512×512 到 1344×768）
3. **输入提示词**：输入正向和反向提示词
4. **设置文件名**：指定输出文件名
5. **开始生成**：显示进度并保存结果

#### 直接生成模式
```bash
python main.py --direct --prompt "your prompt" [OPTIONS]
```

可用选项：
```bash
--prompt TEXT              # 正向提示词（默认："1girl with cat ears..."）
--negative-prompt TEXT     # 反向提示词（默认："low quality..."）
--width INTEGER           # 图像宽度（默认：1024，必须≥512且是64的倍数）
--height INTEGER          # 图像高度（默认：1024，必须≥512且是64的倍数）
--output PATH             # 输出文件路径（默认："generated_images/generated_image.png"）
--model PATH              # 自定义模型路径（默认使用 config 中的模型）
```

**示例**：
```bash
# 简单生成
python main.py --direct --prompt "a beautiful sunset over mountains"

# 完整参数生成
python main.py --direct \
  --prompt "1girl, masterpiece, best quality" \
  --negative-prompt "low quality, blurry" \
  --width 768 \
  --height 768 \
  --output "my_artwork.png"
```

### 图像尺寸建议

| 分辨率 | GPU 显存要求 | 推荐用途 |
|--------|--------------|----------|
| 512×512 | 4GB+ | 快速测试，低配置设备 |
| 768×768 | 8GB+ | 平衡质量与性能（推荐） |
| 1024×1024 | 12GB+ | 高质量输出 |
| 1344×768 | 16GB+ | 宽屏图像 |

**注意**：所有尺寸必须是 64 的倍数，且最小为 512×512。

## ⚙️ 配置说明

### 模型存储位置
- 默认模型目录：`./models/`
- 生成图像目录：`./generated_images/`
- 配置自动创建目录（如果不存在）

### 环境变量
```bash
# 禁用 Hugging Face Hub 的符号链接（Windows 推荐）
export HF_HUB_DISABLE_SYMLINKS=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# PyTorch 相关设置
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # 减少内存碎片
```

### 配置文件
`config.py` 提供以下功能：
- 动态发现 models 目录下的可用模型
- 提供模型路径枚举
- 支持 .safetensors 单文件和 Diffusers 目录格式

## 🧪 测试和验证

项目包含多个测试脚本，用于验证安装和功能：

```bash
# 检查系统资源
python test_simple.py

# 测试模型加载
python test_load.py

# 验证 CUDA 功能
python test_cuda.py

# 最小化图像生成测试
python test_minimal.py

# 调试模型加载
python debug_load.py
```

## 🔍 故障排除

### 常见问题

#### 1. 模型下载失败
```bash
# 检查网络连接
ping modelscope.cn

# 尝试使用镜像源
export MODEL_SCOPE_MIRROR=https://mirror.modelscope.cn
```

#### 2. 显存不足错误
- 降低图像尺寸（如 512×512）
- 确保启用自动优化功能
- 检查是否有其他程序占用 GPU

#### 3. Windows 权限问题
```bash
# 已在代码中自动设置
set HF_HUB_DISABLE_SYMLINKS=1
```

#### 4. Python 导入错误
```bash
# 确保已安装所有依赖
uv sync  # 或 pip install -e .

# 检查 Python 版本
python --version  # 需要 3.10+
```

### 性能优化建议

1. **启用 GPU 加速**：确保 PyTorch 安装正确版本
2. **调整图像尺寸**：根据可用显存选择合适的尺寸
3. **关闭预览**：生成时不显示预览可节省内存
4. **批量处理**：如需生成多张图片，可编写脚本批量处理

## 📚 API 参考

### ImageGenerator 类
```python
from image_generator import ChenkinNoobImageGenerator

# 初始化生成器
generator = ChenkinNoobImageGenerator(model_path="./models/ChenkinNoob")

# 生成图像
output_path = generator.generate_and_save(
    prompt="your prompt",
    negative_prompt="negative prompt",
    width=1024,
    height=1024,
    output_path="output.png",
    num_inference_steps=30,
    guidance_scale=7.5,
    seed=42,
    show_preview=True
)
```

### ModelDownloader 类
```python
from model_downloader import ModelDownloader

# 初始化下载器
downloader = ModelDownloader(cache_dir="./models")

# 下载模型
model_dir = downloader.download(
    model_id="ChenkinNoob/ChenkinNoob-XL-V0.2",
    revision="v1.0",
    ignore_patterns=["*.msgpack"],
    show_progress=True
)
```

## 🤝 贡献指南

我们欢迎各种形式的贡献！

1. **报告问题**：在 GitHub Issues 中报告 bug 或提出建议
2. **提交代码**：通过 Pull Request 提交改进
3. **改进文档**：帮助完善文档或翻译
4. **分享模型**：推荐优秀的开源模型

### 开发环境设置
```bash
# 克隆项目
git clone https://github.com/MeowLiu/open-models.git
cd open-models

# 安装开发依赖
uv sync --dev

# 运行测试
pytest tests/
```

### 代码规范
- 使用 Ruff 进行代码格式化
- 使用 isort 进行导入排序
- 遵循 PEP 8 规范
- 添加适当的文档字符串

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- **ModelScope**：提供优秀的中文模型平台
- **Hugging Face**：提供强大的 transformers 和 diffusers 库
- **Stable Diffusion**：革命性的图像生成技术
- **ChenkinNoob**：优秀的 Stable Diffusion XL 中文模型

## 📞 联系和支持

- **GitHub Issues**：报告问题或提出功能请求
- **文档**：查看本 README 和代码注释
- **社区**：欢迎加入讨论和分享作品

---

**Happy Generating! 🎨**

*如果这个项目对您有帮助，请考虑给它一个 ⭐ 星标支持！*

