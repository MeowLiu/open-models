#!/usr/bin/env python3
"""
ChenkinNoob 图像生成工具 - 主入口点。

提供两种使用方式：
1. 交互式CLI模式（默认）：python main.py
2. 直接生成模式：python main.py --direct --prompt "your prompt"

基于 Stable Diffusion XL 和 InquirerPy+rich 库。
"""

import argparse
import sys
import os

# 禁用 Hugging Face Hub 的符号链接，避免 Windows 权限问题
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from rich import print as rprint


def main():
    """主函数，解析命令行参数并选择运行模式。"""
    parser = argparse.ArgumentParser(
        description="ChenkinNoob 图像生成工具",
        epilog="示例:\n"
               "  python main.py                     # 交互式CLI模式\n"
               "  python main.py --direct            # 直接生成模式（使用默认参数）\n"
               "  python main.py --direct --prompt \"cat girl\" --width 1024 --height 1024",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--direct",
        action="store_true",
        help="直接生成模式（非交互式）"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="1girl with cat ears，masterpiece, best quality, cinematic lighting",
        help="正向提示词（直接模式使用）"
    )

    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="low quality, worst quality, blurry, deformed",
        help="反向提示词（直接模式使用）"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="图像宽度（直接模式使用）"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="图像高度（直接模式使用）"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="generated_images/generated_image.png",
        help="输出文件路径（直接模式使用）"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="模型路径（直接模式使用），默认使用config中的模型"
    )

    args = parser.parse_args()

    if args.direct:
        # 直接生成模式
        rprint("[bold cyan]直接生成模式[/bold cyan]")

        try:
            from image_generator import ChenkinNoobImageGenerator
            from config import ModelsPath

            # 确定模型路径
            if args.model:
                model_path = args.model
            else:
                model_path = ModelsPath.ChenkinNoob.value

            # 验证尺寸
            if args.width < 512 or args.height < 512:
                rprint(f"[red]错误: 图像尺寸必须≥512x512，当前为{args.width}x{args.height}[/red]")
                sys.exit(1)
            if args.width % 64 != 0 or args.height % 64 != 0:
                rprint(f"[red]错误: 图像尺寸必须是64的倍数，当前为{args.width}x{args.height}[/red]")
                sys.exit(1)

            # 确保输出目录存在
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            rprint(f"   模型: {model_path}")
            rprint(f"   尺寸: {args.width}×{args.height}")
            rprint(f"   正向提示词: {args.prompt[:50]}...")
            rprint(f"   输出: {args.output}")

            # 生成图像
            generator = ChenkinNoobImageGenerator(model_path)
            output_path = generator.generate_and_save(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                width=args.width,
                height=args.height,
                output_path=args.output,
                num_inference_steps=30,
                guidance_scale=7.5,
                seed=42,
                show_preview=False,
            )

            rprint(f"[bold green]✓ 图像已成功保存至: {output_path}[/bold green]")

        except Exception as e:
            rprint(f"[red]错误: {e}[/red]")
            sys.exit(1)
    else:
        # 交互式CLI模式（默认）
        try:
            from cli import interactive_cli
            interactive_cli()
        except KeyboardInterrupt:
            rprint("\n[yellow]已取消操作[/yellow]")
            sys.exit(0)
        except ImportError as e:
            rprint(f"[red]导入错误: {e}[/red]")
            rprint("[yellow]请确保已安装所需依赖：pip install inquirerpy rich[/yellow]")
            sys.exit(1)


if __name__ == "__main__":
    main()
