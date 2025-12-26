from enum import Enum
import os
from typing import List, Dict, Optional


# Keep the original enum for backward compatibility if needed
class ModelsPath(str, Enum):
    # 使用绝对路径，避免路径识别问题
    ChenkinNoob = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "models/ChenkinNoob/ChenkinNoob-XL-V0___2"
    ))


def discover_models(models_dir: str = "models") -> Dict[str, str]:
    """
    动态发现models目录下的可用模型。

    Args:
        models_dir: 模型目录路径

    Returns:
        字典，键为模型显示名称，值为模型路径
    """
    models = {}
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), models_dir))

    if not os.path.exists(base_path):
        return models

    # 使用os.walk递归搜索所有子目录
    for root, _dirs, files in os.walk(base_path):
        # 检查当前目录是否有.safetensors文件
        safetensors_files = [f for f in files if f.endswith('.safetensors')]
        has_model_index = 'model_index.json' in files

        if safetensors_files or has_model_index:
            # 计算相对于base_path的相对路径作为显示名称
            rel_path = os.path.relpath(root, base_path)
            # 如果相对路径是当前目录'.'，使用父目录名
            if rel_path == '.':
                # 使用base_path的直接子目录名
                parent_name = os.path.basename(os.path.dirname(root))
                if parent_name == os.path.basename(base_path):
                    # 如果父目录就是models目录本身，使用目录名
                    dir_name = os.path.basename(root)
                    display_name = dir_name if dir_name else 'unknown'
                else:
                    display_name = parent_name
            else:
                # 将路径分隔符替换为斜杠
                display_name = rel_path.replace('\\', '/')

            models[display_name] = os.path.abspath(root)

    return models


def get_available_models() -> List[str]:
    """
    获取可用模型路径列表，用于InquirerPy选项。

    Returns:
        模型路径列表
    """
    models = discover_models()
    if not models:
        # 如果没有动态发现的模型，返回默认模型
        return [ModelsPath.ChenkinNoob.value]

    return list(models.values())


def get_model_display_names() -> List[str]:
    """
    获取模型显示名称列表，用于InquirerPy选项。

    Returns:
        模型显示名称列表
    """
    models = discover_models()
    if not models:
        # 如果没有动态发现的模型，返回默认模型显示名称
        return ["ChenkinNoob (默认)"]

    return list(models.keys())


def get_model_path_by_display_name(display_name: str) -> Optional[str]:
    """
    根据显示名称获取模型路径。

    Args:
        display_name: 模型显示名称

    Returns:
        模型路径，如果未找到则返回None
    """
    models = discover_models()
    return models.get(display_name)
