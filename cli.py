#!/usr/bin/env python3
"""
交互式CLI图像生成工具。

基于InquirerPy和rich库提供类似Vite的交互式体验。
"""

import os
import re
import sys
import traceback
from typing import Optional, Tuple
from pathlib import Path

# 设置环境变量以避免 prompt_toolkit 在非标准终端中的问题
os.environ.setdefault("PROMPT_TOOLKIT_NO_WIN32_CONSOLE", "1")

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from config import get_model_display_names, get_model_path_by_display_name
from image_generator import ChenkinNoobImageGenerator


console = Console()


def validate_resolution(width: int, height: int) -> Tuple[bool, str]:
    """
    验证分辨率是否合法。

    Args:
        width: 宽度
        height: 高度

    Returns:
        (是否合法, 错误信息)
    """
    if width < 512 or height < 512:
        return False, f"图像尺寸必须≥512x512，当前为{width}x{height}"
    if width % 64 != 0 or height % 64 != 0:
        return False, f"图像尺寸必须是64的倍数，当前为{width}x{height}"
    return True, ""


def check_system_resources() -> Tuple[bool, str]:
    """
    检查系统资源是否足够运行模型。

    Returns:
        (是否足够, 警告信息)
    """
    warnings = []

    try:
        import torch
        import psutil

        # 检查GPU显存
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory_gb < 8:
                warnings.append(
                    f"GPU显存较低 ({gpu_memory_gb:.1f}GB)，建议至少8GB。将使用CPU卸载模式。"
                )
        else:
            warnings.append("未检测到CUDA GPU，将使用CPU模式（速度较慢）。")

        # 检查系统内存
        mem = psutil.virtual_memory()
        available_memory_gb = mem.available / 1024**3
        # 模型大小约6.6GB，加载需要约13-20GB内存
        if available_memory_gb < 12:
            warnings.append(
                f"可用系统内存较低 ({available_memory_gb:.1f}GB)，建议关闭其他应用程序。"
            )

        if warnings:
            return False, "\n".join(warnings)
        return True, ""

    except ImportError:
        # 如果无法导入psutil，跳过内存检查
        return True, ""


def print_resource_warnings():
    """打印资源警告（如果适用）。"""
    is_ok, warning_msg = check_system_resources()
    if not is_ok and warning_msg:
        rprint("\n[yellow]警告: 系统资源可能不足[/yellow]")
        for line in warning_msg.split("\n"):
            rprint(f"  {line}")


def get_next_filename(base_name: str, extension: str = "png") -> str:
    """
    获取下一个可用的文件名（避免冲突）。

    Args:
        base_name: 基础文件名（不含扩展名）
        extension: 文件扩展名

    Returns:
        可用的文件名
    """
    output_dir = Path("generated_images")
    output_dir.mkdir(exist_ok=True)

    # 验证基础文件名
    if not re.match(r"^[a-zA-Z0-9_-]{3,50}$", base_name):
        # 如果不匹配，使用默认名称
        base_name = "generated_image"

    filename = f"{base_name}.{extension}"
    counter = 1

    while (output_dir / filename).exists():
        filename = f"{base_name}_{counter}.{extension}"
        counter += 1

    return str(output_dir / filename)


def interactive_cli() -> None:
    """主交互式CLI函数。"""
    # 显示欢迎信息
    console.clear()
    rprint(
        Panel.fit(
            "[bold cyan]ChenkinNoob 图像生成工具[/bold cyan]\n"
            "基于 Stable Diffusion XL 的交互式图像生成",
            border_style="cyan",
        )
    )

    # 步骤1: 选择模型
    rprint("\n[bold]步骤 1/5: 选择模型[/bold]")
    model_display_names = get_model_display_names()

    if not model_display_names:
        rprint("[red]错误: 未找到任何模型。请先下载模型。[/red]")
        sys.exit(1)

    selected_display_name = inquirer.select(
        message="选择要使用的模型:",
        choices=model_display_names,
        default=model_display_names[0],
    ).execute()

    # 获取模型路径
    model_path = get_model_path_by_display_name(selected_display_name)
    if not model_path:
        # 如果动态发现失败，使用默认模型
        from config import ModelsPath

        model_path = ModelsPath.ChenkinNoob.value
        rprint(f"[yellow]警告: 使用默认模型路径: {model_path}[/yellow]")

    # 步骤2: 选择宽高比
    rprint("\n[bold]步骤 2/5: 选择图像尺寸[/bold]")
    aspect_ratio_choices = [
        Choice(value=(768, 768), name="3:3 (Square): 768×768 (推荐)"),
        Choice(value=(512, 512), name="1:1 (Square): 512×512 (低显存)"),
        Choice(value=(1024, 1024), name="1:1 (Square): 1024×1024 (高显存)"),
        Choice(value=(768, 1024), name="3:4 (Portrait): 768×1024"),
        Choice(value=(768, 1344), name="9:16 (Portrait): 768×1344"),
        Choice(value=(1344, 768), name="16:9 (Landscape): 1344×768"),
        Choice(value="custom", name="自定义尺寸"),
    ]

    aspect_choice = inquirer.select(
        message="选择宽高比:",
        choices=aspect_ratio_choices,
        default=aspect_ratio_choices[0],
    ).execute()

    if aspect_choice == "custom":
        # 自定义尺寸
        while True:
            try:
                width = inquirer.number(
                    message="请输入图像宽度:",
                    min_allowed=512,
                    default=768,
                    validate=lambda val: isinstance(val, (int, float))
                    and int(val) % 64 == 0,
                    invalid_message="宽度必须是64的倍数且≥512",
                ).execute()

                height = inquirer.number(
                    message="请输入图像高度:",
                    min_allowed=512,
                    default=768,
                    validate=lambda val: isinstance(val, (int, float))
                    and int(val) % 64 == 0,
                    invalid_message="高度必须是64的倍数且≥512",
                ).execute()

                width = int(width)
                height = int(height)

                is_valid, error_msg = validate_resolution(width, height)
                if is_valid:
                    break
                else:
                    rprint(f"[red]{error_msg}[/red]")
            except Exception as e:
                rprint(f"[red]输入错误: {e}[/red]")
    else:
        width, height = aspect_choice

    rprint(f"[green]✓ 图像尺寸: {width}×{height}[/green]")

    # 检查系统资源
    print_resource_warnings()

    # 步骤3: 输入正向提示词
    rprint("\n[bold]步骤 3/5: 输入正向提示词[/bold]")
    default_positive_prompt = "1girl with cat ears，masterpiece, best quality, cinematic lighting, soft pastel color palette, ,large sparkling eyes, delicate facial features, serene atmosphere, intricate details"

    positive_prompt = inquirer.text(
        message="正向提示词 (支持多行，Ctrl+Z/Ctrl+D结束):",
        default=default_positive_prompt,
        # multiline=True,
    ).execute()

    # 步骤4: 输入反向提示词
    rprint("\n[bold]步骤 4/5: 输入反向提示词[/bold]")
    default_negative_prompt = "low quality, worst quality, blurry, deformed, disfigured, extra limbs, fused fingers, bad anatomy, text, signature, watermark, username, ugly, duplicate, morbid, mutilated, anime screentone patterns, jpeg artifacts, out of focus, monochrome, grayscale, doll-like face"

    negative_prompt = inquirer.text(
        message="反向提示词 (支持多行，Ctrl+Z/Ctrl+D结束):",
        default=default_negative_prompt,
        # multiline=True,
    ).execute()

    # 步骤5: 输入文件名
    rprint("\n[bold]步骤 5/5: 输入文件名[/bold]")
    while True:
        filename = inquirer.text(
            message="文件名 (仅字母、数字、下划线、连字符，3-50字符):",
            default="generated_image",
            validate=lambda val: re.match(r"^[a-zA-Z0-9_-]{3,50}$", val) is not None,
            invalid_message="文件名必须仅包含字母、数字、下划线、连字符，长度3-50字符",
        ).execute()

        output_path = get_next_filename(filename)
        if Path(output_path).exists():
            # 实际上get_next_filename已经处理了冲突，但这里再次确认
            rprint(f"[yellow]文件已存在，将保存为: {output_path}[/yellow]")
            break
        else:
            break

    # 确认信息
    rprint("\n[bold cyan]✓ 配置完成，开始生成图像[/bold cyan]")
    rprint(f"   模型: {selected_display_name}")
    rprint(f"   尺寸: {width}×{height}")
    rprint(f"   输出: {output_path}")
    rprint(f"   正向提示词长度: {len(positive_prompt)} 字符")
    rprint(f"   反向提示词长度: {len(negative_prompt)} 字符")

    # 确认是否继续
    confirm = inquirer.confirm(
        message="是否开始生成图像?",
        default=True,
    ).execute()

    if not confirm:
        rprint("[yellow]已取消生成[/yellow]")
        sys.exit(0)

    # 初始化图像生成器
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]正在加载模型...", total=None)

            generator = ChenkinNoobImageGenerator(model_path)
            progress.update(task, description="[cyan]正在生成图像...")
            from random import randint

            # 生成图像
            output_path = generator.generate_and_save(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                output_path=output_path,
                num_inference_steps=30,
                guidance_scale=7.5,
                seed=randint(1, 100),
                show_preview=False,  # 不在CLI中显示预览
            )

            progress.update(task, description="[green]✓ 图像生成完成![/green]")

        rprint(
            f"\n[bold green]✓ 图像已成功保存至:[/bold green] [cyan]{output_path}[/cyan]"
        )

        # 询问是否显示预览
        show_preview = inquirer.confirm(
            message="是否显示图像预览?",
            default=False,
        ).execute()

        if show_preview:
            import matplotlib.pyplot as plt
            from PIL import Image

            image = Image.open(output_path)
            plt.imshow(image)
            plt.axis("off")
            plt.show()

        # 询问是否继续生成
        continue_generation = inquirer.confirm(
            message="是否继续生成另一张图像?",
            default=False,
        ).execute()

        if continue_generation:
            interactive_cli()

    except FileNotFoundError as e:
        rprint(f"[red]错误: 模型文件未找到[/red]\n{e}")
        available_models = get_model_display_names()
        if available_models:
            rprint("[yellow]可用模型:[/yellow]")
            for model in available_models:
                rprint(f"  - {model}")
        sys.exit(1)
    except ValueError as e:
        rprint(f"[red]错误: 无效的参数[/red]\n{e}")
        sys.exit(1)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            rprint("[red]错误: 显存不足[/red]")
            rprint("[yellow]建议:[/yellow]")
            rprint("  1. 降低图像分辨率（如768×768）")
            rprint("  2. 关闭其他占用显存的应用程序")
            rprint("  3. 使用CPU模式（速度较慢）")
        else:
            rprint(f"[red]运行时错误:[/red]\n{e}")
        sys.exit(1)
    except Exception as e:
        rprint(f"[red]未知错误:[/red]\n{e}")
        rprint(f"[yellow]详细错误信息:[/yellow]")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        interactive_cli()
    except KeyboardInterrupt:
        rprint("\n[yellow]已取消操作[/yellow]")
        sys.exit(0)
