from enum import Enum
import os


# Keep the original enum for backward compatibility if needed
class ModelsPath(str, Enum):
    # 使用绝对路径，避免路径识别问题
    ChenkinNoob = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "models/ChenkinNoob/ChenkinNoob-XL-V0___2"
    ))
