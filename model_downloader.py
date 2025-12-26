 
"""
ModelScope Model Downloader - A CLI tool for downloading and managing models from ModelScope community.

This tool provides a simple command-line interface to download models from
ModelScope (https://modelscope.cn). It includes features like:
- Download models with various options (revision, ignore patterns)
- Check if models already exist locally to avoid re-downloading
- List locally available models with size information
- Force re-download if needed

Basic Usage:
    python model_downloader.py [MODEL_ID] [OPTIONS]
    python model_downloader.py --list
    python model_downloader.py --help

Examples:
    # Download a model
    python model_downloader.py ChenkinNoob/ChenkinNoob-XL-V0.2

    # List locally available models
    python model_downloader.py --list

    # Download with specific revision and ignore patterns
    python model_downloader.py ChenkinNoob/ChenkinNoob-XL-V0.2 --revision v1.0 --ignore-pattern "*.msgpack"

    # Force re-download even if model exists
    python model_downloader.py ChenkinNoob/ChenkinNoob-XL-V0.2 --force

    # Use custom cache directory
    python model_downloader.py ChenkinNoob/ChenkinNoob-XL-V0.2 --cache-dir /path/to/models

The ModelDownloader class can also be imported and used programmatically.

Author: MeowLiu
Date: 2025-12-26
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List
from modelscope import snapshot_download


class ModelDownloader:
    """A class for downloading models from ModelScope community."""

    def __init__(self, cache_dir: str = "./models"):
        """
        Initialize the downloader.

        Args:
            cache_dir: Directory to store downloaded models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self, model_id: str, revision: Optional[str] = None,
                 ignore_patterns: Optional[List[str]] = None,
                 show_progress: bool = True) -> str:
        """
        Download a model from ModelScope.

        Args:
            model_id: Model ID in format 'username/model_name'
            revision: Specific revision or branch to download
            ignore_patterns: List of file patterns to ignore
            show_progress: Whether to show download progress

        Returns:
            Path to the downloaded model directory
        """
        if not self._validate_model_id(model_id):
            raise ValueError(f"Invalid model ID format: {model_id}. Expected 'username/model_name'")

        if show_progress:
            print(f"\n[DOWNLOAD] Downloading model: {model_id}")
            if revision:
                print(f"   Revision: {revision}")
            if ignore_patterns:
                print(f"   Ignoring patterns: {ignore_patterns}")
            print("-" * 50)

        # Download the model
        model_dir = snapshot_download(
            model_id,
            cache_dir=str(self.cache_dir),
            revision=revision,
            ignore_file_pattern=ignore_patterns
        )

        if show_progress:
            print("\n[SUCCESS] Model downloaded successfully!")
            print(f"   Location: {model_dir}")
            print("-" * 50)

        return model_dir

    def list_models(self) -> List[dict]:
        """
        List all models in the cache directory.

        Returns:
            List of dictionaries with model info
        """
        models = []
        if self.cache_dir.exists():
            for item in self.cache_dir.iterdir():
                # Skip hidden directories (starting with .) and non-directories
                if item.is_dir() and not item.name.startswith('.'):
                    models.append({
                        'name': item.name,
                        'path': str(item.absolute()),
                        'size': self._get_directory_size(item)
                    })
        return models

    def model_exists(self, model_id: str) -> bool:
        """
        Check if a model already exists locally.

        Args:
            model_id: Model ID to check

        Returns:
            True if model exists, False otherwise
        """
        model_name = self._extract_model_name(model_id)
        model_path = self.cache_dir / model_name
        return model_path.exists()

    def get_model_path(self, model_id: str) -> Optional[Path]:
        """
        Get local path for a model if it exists.

        Args:
            model_id: Model ID to look for

        Returns:
            Path to model directory or None
        """
        model_name = self._extract_model_name(model_id)
        model_path = self.cache_dir / model_name
        if model_path.exists():
            return model_path
        return None

    def _validate_model_id(self, model_id: str) -> bool:
        """Validate model ID format."""
        if not model_id or '/' not in model_id:
            return False
        parts = model_id.split('/')
        return len(parts) == 2 and all(parts)

    def _extract_model_name(self, model_id: str) -> str:
        """Extract model name from model ID."""
        return model_id.split('/')[-1]

    def _get_directory_size(self, directory: Path) -> int:
        """Calculate directory size in bytes."""
        total_size = 0
        for dirpath, _, filenames in os.walk(directory):
            for filename in filenames:
                file_path = Path(dirpath) / filename
                if file_path.exists():
                    total_size += file_path.stat().st_size
        return total_size


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download models from ModelScope community",
        epilog="Example: python model_downloader.py ChenkinNoob/ChenkinNoob-XL-V0.2"
    )

    parser.add_argument(
        "model_id",
        nargs="?",
        help="Model ID to download (format: username/model_name)"
    )

    parser.add_argument(
        "--cache-dir",
        default="./models",
        help="Directory to store downloaded models (default: ./models)"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List locally available models"
    )

    parser.add_argument(
        "--revision",
        help="Specific revision or branch to download"
    )

    parser.add_argument(
        "--ignore-pattern",
        action="append",
        help="File patterns to ignore (e.g., '*.msgpack', '*.onnx')"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if model exists locally"
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = ModelDownloader(cache_dir=args.cache_dir)

    # Handle --list option
    if args.list:
        models = downloader.list_models()
        if models:
            print("\n[DIR] Available models in cache directory:")
            print("-" * 60)
            print(f"{'Model Name':<30} {'Size':<12} {'Location':<30}")
            print("-" * 60)
            for model in models:
                size_mb = model['size'] / (1024 * 1024)
                if size_mb < 1024:
                    size_str = f"{size_mb:.2f} MB"
                else:
                    size_gb = size_mb / 1024
                    size_str = f"{size_gb:.2f} GB"
                # Truncate path for display
                path_display = model['path']
                if len(path_display) > 30:
                    path_display = "..." + path_display[-27:]
                print(f"{model['name']:<30} {size_str:<12} {path_display:<30}")
            print("-" * 60)
            print(f"Total: {len(models)} model(s)")
        else:
            print("\n[EMPTY] No models found in cache directory.")
            print(f"   Cache directory: {downloader.cache_dir.absolute()}")
        return

    # Handle model download
    if args.model_id:
        # Validate model ID
        if not downloader._validate_model_id(args.model_id):
            print(f"Error: Invalid model ID format: {args.model_id}")
            print("Expected format: username/model_name")
            sys.exit(1)

        # Check if model exists
        if downloader.model_exists(args.model_id) and not args.force:
            print(f"\n[EXISTS] Model '{args.model_id}' already exists locally.")
            model_path = downloader.get_model_path(args.model_id)
            if model_path:
                size = downloader._get_directory_size(model_path) / (1024 * 1024)
                if size < 1024:
                    size_str = f"{size:.2f} MB"
                else:
                    size_str = f"{size/1024:.2f} GB"
                print(f"   Location: {model_path}")
                print(f"   Size: {size_str}")
            print("   Use --force flag to re-download.")
            return

        # Download the model
        try:
            model_dir = downloader.download(
                args.model_id,
                revision=args.revision,
                ignore_patterns=args.ignore_pattern,
                show_progress=True
            )
            # Show final summary
            if model_dir:
                final_path = Path(model_dir)
                if final_path.exists():
                    size = downloader._get_directory_size(final_path) / (1024 * 1024)
                    if size < 1024:
                        size_str = f"{size:.2f} MB"
                    else:
                        size_str = f"{size/1024:.2f} GB"
                    print("\n[SUMMARY] Download Summary:")
                    print(f"   Model: {args.model_id}")
                    print(f"   Location: {model_dir}")
                    print(f"   Size: {size_str}")
                    print(f"   Cache directory: {downloader.cache_dir.absolute()}")
        except Exception as e:
            print(f"\n[ERROR] Error downloading model: {e}")
            sys.exit(1)
    else:
        # No arguments provided
        parser.print_help()


if __name__ == "__main__":
    main()