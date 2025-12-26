 
"""
ModelScope 模型下载器 - 一个用于从 ModelScope 社区下载和管理模型的命令行工具。

此工具提供了一个简单的命令行界面，用于从 ModelScope (https://modelscope.cn) 下载模型。它包含以下功能：
- 使用各种选项下载模型（版本、忽略模式）
- 检查模型是否已本地存在以避免重新下载
- 列出本地可用的模型及其大小信息
- 如需要强制重新下载

基本用法：
    python model_downloader.py [MODEL_ID] [OPTIONS]
    python model_downloader.py --list
    python model_downloader.py --help

示例：
    # 下载模型
    python model_downloader.py ChenkinNoob/ChenkinNoob-XL-V0.2

    # 列出本地可用的模型
    python model_downloader.py --list

    # 使用特定版本和忽略模式下载
    python model_downloader.py ChenkinNoob/ChenkinNoob-XL-V0.2 --revision v1.0 --ignore-pattern "*.msgpack"

    # 即使模型存在也强制重新下载
    python model_downloader.py ChenkinNoob/ChenkinNoob-XL-V0.2 --force

    # 使用自定义缓存目录
    python model_downloader.py ChenkinNoob/ChenkinNoob-XL-V0.2 --cache-dir /path/to/models

ModelDownloader 类也可以被导入并在程序中使用。

作者: MeowLiu
日期: 2025-12-26
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List
from modelscope import snapshot_download


class ModelDownloader:
    """一个用于从 ModelScope 社区下载模型的类。"""

    def __init__(self, cache_dir: str = "./models"):
        """
        初始化下载器。

        参数:
            cache_dir: 存储下载模型的目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self, model_id: str, revision: Optional[str] = None,
                 ignore_patterns: Optional[List[str]] = None,
                 show_progress: bool = True) -> str:
        """
        从 ModelScope 下载模型。

        参数:
            model_id: 模型 ID，格式为 'username/model_name'
            revision: 要下载的特定版本或分支
            ignore_patterns: 要忽略的文件模式列表
            show_progress: 是否显示下载进度

        返回:
            下载模型目录的路径
        """
        if not self._validate_model_id(model_id):
            raise ValueError(f"无效的模型 ID 格式: {model_id}. 预期格式 'username/model_name'")

        if show_progress:
            print(f"\n[DOWNLOAD] 正在下载模型: {model_id}")
            if revision:
                print(f"   版本: {revision}")
            if ignore_patterns:
                print(f"   忽略模式: {ignore_patterns}")
            print("-" * 50)

        # Download the model
        model_dir = snapshot_download(
            model_id,
            cache_dir=str(self.cache_dir),
            revision=revision,
            ignore_file_pattern=ignore_patterns
        )

        if show_progress:
            print("\n[SUCCESS] 模型下载成功!")
            print(f"   位置: {model_dir}")
            print("-" * 50)

        return model_dir

    def list_models(self) -> List[dict]:
        """
        列出缓存目录中的所有模型。

        返回:
            包含模型信息的字典列表
        """
        models = []
        if self.cache_dir.exists():
            for item in self.cache_dir.iterdir():
                # 跳过隐藏目录（以.开头）和非目录
                if item.is_dir() and not item.name.startswith('.'):
                    models.append({
                        'name': item.name,
                        'path': str(item.absolute()),
                        'size': self._get_directory_size(item)
                    })
        return models

    def model_exists(self, model_id: str) -> bool:
        """
        检查模型是否已本地存在。

        参数:
            model_id: 要检查的模型 ID

        返回:
            如果模型存在则返回 True，否则返回 False
        """
        model_name = self._extract_model_name(model_id)
        model_path = self.cache_dir / model_name
        return model_path.exists()

    def get_model_path(self, model_id: str) -> Optional[Path]:
        """
        获取模型的本地路径（如果存在）。

        参数:
            model_id: 要查找的模型 ID

        返回:
            模型目录的路径或 None
        """
        model_name = self._extract_model_name(model_id)
        model_path = self.cache_dir / model_name
        if model_path.exists():
            return model_path
        return None

    def _validate_model_id(self, model_id: str) -> bool:
        """验证模型 ID 格式。"""
        if not model_id or '/' not in model_id:
            return False
        parts = model_id.split('/')
        return len(parts) == 2 and all(parts)

    def _extract_model_name(self, model_id: str) -> str:
        """从模型 ID 中提取模型名称。"""
        return model_id.split('/')[-1]

    def _get_directory_size(self, directory: Path) -> int:
        """计算目录大小（以字节为单位）。"""
        total_size = 0
        for dirpath, _, filenames in os.walk(directory):
            for filename in filenames:
                file_path = Path(dirpath) / filename
                if file_path.exists():
                    total_size += file_path.stat().st_size
        return total_size


def main():
    """主命令行入口点。"""
    parser = argparse.ArgumentParser(
        description="从 ModelScope 社区下载模型",
        epilog="示例: python model_downloader.py ChenkinNoob/ChenkinNoob-XL-V0.2"
    )

    parser.add_argument(
        "model_id",
        nargs="?",
        help="要下载的模型 ID（格式: username/model_name）"
    )

    parser.add_argument(
        "--cache-dir",
        default="./models",
        help="存储下载模型的目录（默认: ./models）"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="列出本地可用的模型"
    )

    parser.add_argument(
        "--revision",
        help="要下载的特定版本或分支"
    )

    parser.add_argument(
        "--ignore-pattern",
        action="append",
        help="要忽略的文件模式（例如: '*.msgpack', '*.onnx'）"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="即使模型在本地存在也强制下载"
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = ModelDownloader(cache_dir=args.cache_dir)

    # Handle --list option
    if args.list:
        models = downloader.list_models()
        if models:
            print("\n[DIR] 缓存目录中的可用模型:")
            print("-" * 60)
            print(f"{'模型名称':<30} {'大小':<12} {'位置':<30}")
            print("-" * 60)
            for model in models:
                size_mb = model['size'] / (1024 * 1024)
                if size_mb < 1024:
                    size_str = f"{size_mb:.2f} MB"
                else:
                    size_gb = size_mb / 1024
                    size_str = f"{size_gb:.2f} GB"
                # 截断路径以供显示
                path_display = model['path']
                if len(path_display) > 30:
                    path_display = "..." + path_display[-27:]
                print(f"{model['name']:<30} {size_str:<12} {path_display:<30}")
            print("-" * 60)
            print(f"总计: {len(models)} 个模型")
        else:
            print("\n[EMPTY] 在缓存目录中未找到模型.")
            print(f"   缓存目录: {downloader.cache_dir.absolute()}")
        return

    # Handle model download
    if args.model_id:
        # 验证模型 ID
        if not downloader._validate_model_id(args.model_id):
            print(f"错误: 无效的模型 ID 格式: {args.model_id}")
            print("预期格式: username/model_name")
            sys.exit(1)

        # 检查模型是否存在
        if downloader.model_exists(args.model_id) and not args.force:
            print(f"\n[EXISTS] 模型 '{args.model_id}' 已在本地存在.")
            model_path = downloader.get_model_path(args.model_id)
            if model_path:
                size = downloader._get_directory_size(model_path) / (1024 * 1024)
                if size < 1024:
                    size_str = f"{size:.2f} MB"
                else:
                    size_str = f"{size/1024:.2f} GB"
                print(f"   位置: {model_path}")
                print(f"   大小: {size_str}")
            print("   使用 --force 标志重新下载.")
            return

        # 下载模型
        try:
            model_dir = downloader.download(
                args.model_id,
                revision=args.revision,
                ignore_patterns=args.ignore_pattern,
                show_progress=True
            )
            # 显示最终摘要
            if model_dir:
                final_path = Path(model_dir)
                if final_path.exists():
                    size = downloader._get_directory_size(final_path) / (1024 * 1024)
                    if size < 1024:
                        size_str = f"{size:.2f} MB"
                    else:
                        size_str = f"{size/1024:.2f} GB"
                    print("\n[SUMMARY] 下载摘要:")
                    print(f"   模型: {args.model_id}")
                    print(f"   位置: {model_dir}")
                    print(f"   大小: {size_str}")
                    print(f"   缓存目录: {downloader.cache_dir.absolute()}")
        except Exception as e:
            print(f"\n[ERROR] 下载模型时出错: {e}")
            sys.exit(1)
    else:
        # 未提供参数
        parser.print_help()


if __name__ == "__main__":
    main()