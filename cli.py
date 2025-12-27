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


def get_sequential_filename(base_name: str, index: int, extension: str = "png") -> str:
    """
    获取带有序号的文件名（不检查是否存在）。

    Args:
        base_name: 基础文件名（不含扩展名）
        index: 序号（从1开始）
        extension: 文件扩展名

    Returns:
        带有序号的文件路径
    """
    output_dir = Path("generated_images")
    output_dir.mkdir(exist_ok=True)

    # 验证基础文件名
    if not re.match(r"^[a-zA-Z0-9_-]{3,50}$", base_name):
        # 如果不匹配，使用默认名称
        base_name = "generated_image"

    if index <= 0:
        index = 1

    if index == 1:
        filename = f"{base_name}.{extension}"
    else:
        filename = f"{base_name}_{index}.{extension}"

    return str(output_dir / filename)


def generate_sequential_filenames(base_name: str, count: int, extension: str = "png") -> list[str]:
    """
    生成多个顺序文件名，跳过已存在的文件。

    Args:
        base_name: 基础文件名（不含扩展名）
        count: 需要生成的文件名数量
        extension: 文件扩展名

    Returns:
        文件名路径列表
    """
    filenames = []
    output_dir = Path("generated_images")
    output_dir.mkdir(exist_ok=True)

    # 验证基础文件名
    if not re.match(r"^[a-zA-Z0-9_-]{3,50}$", base_name):
        base_name = "generated_image"

    index = 1
    while len(filenames) < count:
        if index == 1:
            candidate = output_dir / f"{base_name}.{extension}"
        else:
            candidate = output_dir / f"{base_name}_{index}.{extension}"

        if not candidate.exists():
            filenames.append(str(candidate))

        index += 1

    return filenames


def single_generation_session() -> str:
    """
    单次图像生成会话。

    Returns:
        str: 会话状态，可能的值：
            "continue" - 用户希望继续生成另一张图像
            "completed" - 用户完成生成
            "cancelled" - 用户取消生成
            "error" - 发生错误，无法继续
    """
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
                    message="请输入图像宽度（建议输入64的倍数）:",
                    min_allowed=512,
                    default=768,
                    invalid_message="宽度必须是64的倍数且≥512",
                ).execute()

                height = inquirer.number(
                    message="请输入图像高度（建议输入64的倍数）:",
                    min_allowed=512,
                    default=768,
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
    default_positive_prompt = (
        "masterpiece, best quality, 1girl, beautiful detailed eyes, anime style"
    )

    positive_prompt = inquirer.text(
        message="正向提示词 (支持多行，Ctrl+Z/Ctrl+D结束):",
        default=default_positive_prompt,
        # multiline=True,
    ).execute()

    # 步骤4: 输入反向提示词
    rprint("\n[bold]步骤 4/5: 输入反向提示词[/bold]")
    default_negative_prompt = "low quality, worst quality, blurry, deformed"

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

    # 选择生成模式
    mode_choice = inquirer.select(
        message="选择生成模式:",
        choices=[
            Choice(value="single", name="单张生成"),
            Choice(value="multiple", name="多张生成（批量）"),
        ],
        default="single",
    ).execute()

    image_count = 1
    if mode_choice == "multiple":
        while True:
            try:
                count_input = inquirer.number(
                    message="请输入生成图像数量 (1-100):",
                    min_allowed=1,
                    max_allowed=100,
                    default=5,
                ).execute()
                image_count = int(count_input)
                break
            except Exception as e:
                rprint(f"[red]输入错误: {e}[/red]")

    # 确认是否继续
    confirm = inquirer.confirm(
        message=f"是否开始生成{'这1张图像' if mode_choice == 'single' else f'这{image_count}张图像'}?",
        default=True,
    ).execute()

    if not confirm:
        rprint("[yellow]已取消生成[/yellow]")
        return "cancelled"  # 用户取消，未完成生成

    # 初始化图像生成器
    try:
        from random import randint

        if mode_choice == "single":
            # 单张生成模式
            # 初始化生成器
            generator = ChenkinNoobImageGenerator(model_path)

            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("[cyan]正在生成图像...", total=None)

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

                # 清理显存
                generator.cleanup()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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
                    return "continue"  # 用户希望继续生成另一张图像
                else:
                    return "completed"  # 用户完成生成

            except Exception:
                # 发生异常时也清理生成器
                generator.cleanup()
                raise
        else:
            # 多张生成模式
            generator = None
            try:
                # 生成多个文件名
                base_filename = filename  # 从之前的输入获取基础文件名
                output_paths = generate_sequential_filenames(base_filename, image_count)

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("[cyan]正在加载模型...", total=None)

                    # 初始化生成器
                    generator = ChenkinNoobImageGenerator(model_path)

                    # 更新进度条显示为多图生成
                    progress.update(task, description=f"[cyan]正在批量生成 {image_count} 张图像...")

                    generated_count = 0
                    for i, current_output_path in enumerate(output_paths):
                        # 生成随机种子
                        seed = randint(1, 10000)  # 使用更大的随机范围

                        # 更新进度条显示当前进度
                        progress.update(task, description=f"[cyan]正在生成第 {i+1}/{image_count} 张图像...")

                        # 生成图像
                        generator.generate_and_save(
                            prompt=positive_prompt,
                            negative_prompt=negative_prompt,
                            width=width,
                            height=height,
                            output_path=current_output_path,
                            num_inference_steps=30,
                            guidance_scale=7.5,
                            seed=seed,
                            show_preview=False,  # 多图生成时不显示预览
                        )
                        generated_count += 1

                        # 每生成5张图像或最后一张时，强制清理显存
                        if generated_count % 5 == 0 or generated_count == image_count:
                            # 清理显存
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                    progress.update(task, description=f"[green]✓ 已成功生成 {image_count} 张图像![/green]")

                # 显示生成结果摘要
                rprint(f"\n[bold green]✓ 批量生成完成！共生成 {image_count} 张图像:[/bold green]")
                for i, path in enumerate(output_paths[:10]):  # 最多显示前10个文件
                    rprint(f"  [{i+1}] [cyan]{Path(path).name}[/cyan]")

                if image_count > 10:
                    rprint(f"  ... 还有 {image_count - 10} 张图像未显示")

                rprint(f"[green]所有图像已保存至:[/green] [cyan]{Path('generated_images').absolute()}[/cyan]")

                # 多图生成后不询问是否显示预览，直接询问是否继续
                continue_generation = inquirer.confirm(
                    message="是否继续生成另一批图像?",
                    default=False,
                ).execute()

                # 清理生成器资源
                if generator:
                    generator.cleanup()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if continue_generation:
                    return "continue"  # 用户希望继续生成另一批图像
                else:
                    return "completed"  # 用户完成生成

            finally:
                # 确保在异常情况下也清理资源
                if generator:
                    generator.cleanup()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    except FileNotFoundError as e:
        rprint(f"[red]错误: 模型文件未找到[/red]\n{e}")
        available_models = get_model_display_names()
        if available_models:
            rprint("[yellow]可用模型:[/yellow]")
            for model in available_models:
                rprint(f"  - {model}")
        return "error"  # 模型文件未找到错误
    except ValueError as e:
        rprint(f"[red]错误: 无效的参数[/red]\n{e}")
        return "error"  # 参数错误
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            rprint("[red]错误: 显存不足[/red]")
            rprint("[yellow]建议:[/yellow]")
            rprint("  1. 降低图像分辨率（如768×768）")
            rprint("  2. 关闭其他占用显存的应用程序")
            rprint("  3. 使用CPU模式（速度较慢）")
        else:
            rprint(f"[red]运行时错误:[/red]\n{e}")
        return "error"  # 运行时错误
    except Exception as e:
        rprint(f"[red]未知错误:[/red]\n{e}")
        rprint(f"[yellow]详细错误信息:[/yellow]")
        traceback.print_exc()
        return "error"  # 未知错误


def interactive_cli() -> None:
    """
    交互式CLI主入口点。

    管理多次图像生成会话，避免内存泄漏。
    """
    generation_count = 0
    max_generations = 10  # 安全限制，防止无限循环

    rprint("[cyan]欢迎使用 ChenkinNoob 图像生成工具[/cyan]")
    rprint("[yellow]提示: 每次生成后，程序会询问是否继续生成另一张图像。[/yellow]")
    rprint("[yellow]       每次生成都是独立的会话，避免内存泄漏。[/yellow]")

    while generation_count < max_generations:
        generation_count += 1
        rprint(f"\n[bold]第 {generation_count} 次生成会话[/bold]")

        try:
            status = single_generation_session()

            if status == "continue":
                rprint("[yellow]准备开始下一次生成会话...[/yellow]")
                continue
            elif status == "completed":
                rprint("[green]✓ 图像生成完成，退出程序。[/green]")
                break
            elif status == "cancelled":
                rprint("[yellow]已取消生成。[/yellow]")
                break
            elif status == "error":
                rprint("[red]发生错误，程序将退出。[/red]")
                break
            else:
                rprint(f"[yellow]未知状态: {status}，程序将退出。[/yellow]")
                break

        except KeyboardInterrupt:
            rprint("\n[yellow]已取消操作[/yellow]")
            break
        except Exception as e:
            rprint(f"[red]主循环发生未预期的错误: {e}[/red]")
            traceback.print_exc()
            break

    if generation_count >= max_generations:
        rprint(f"[yellow]已达到最大生成次数限制 ({max_generations} 次)，程序退出。[/yellow]")
        rprint("[yellow]这是为了防止无限循环和内存泄漏。[/yellow]")

    rprint("\n[cyan]感谢使用 ChenkinNoob 图像生成工具![/cyan]")


if __name__ == "__main__":
    interactive_cli()
