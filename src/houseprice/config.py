"""全局配置：路径、随机种子、切分参数。

所有可变量集中在这里，训练脚本与服务端共用同一份定义，
避免出现「训练用一套参数、推理用另一套」的隐性偏差。
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = PROJECT_ROOT / "docs" / "figures"

# 允许通过环境变量覆盖，便于测试与容器化部署
MODELS_DIR: Path = Path(os.environ.get("HOUSEPRICE_MODELS_DIR", PROJECT_ROOT / "models"))

TARGET: str = "SalePrice"
OPENML_NAME: str = "house_prices"
OPENML_VERSION: int = 1

SEED: int = 42
N_SPLITS: int = 5
TEST_SIZE: float = 0.2

# 80% 预测区间的两个分位点
QUANTILES: tuple[float, float] = (0.1, 0.9)

# 缺失率超过该比例的列直接剔除
MAX_MISSING_RATIO: float = 0.45


def ensure_dirs() -> None:
    """创建运行期需要的目录。"""
    for path in (DATA_DIR, RAW_DIR, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
