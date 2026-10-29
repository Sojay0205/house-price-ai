"""测试夹具：用合成数据构造轻量模型，测试全程离线、秒级完成。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from houseprice.data.loader import default_values, schema
from houseprice.data.preprocess import build_pipeline, coerce_types
from houseprice.features.build import add_engineered_features
from houseprice.models import registry


@pytest.fixture
def toy_frame() -> tuple[pd.DataFrame, pd.Series]:
    """12 列数值 + 2 列类别的合成数据集，结构与 Ames 的关键字段一致。"""
    rng = np.random.default_rng(7)
    n = 120
    df = pd.DataFrame(
        {
            "OverallQual": rng.integers(1, 11, n),
            "GrLivArea": rng.integers(500, 3000, n),
            "TotalBsmtSF": rng.integers(0, 1500, n),
            "1stFlrSF": rng.integers(400, 2000, n),
            "2ndFlrSF": rng.integers(0, 1000, n),
            "FullBath": rng.integers(0, 4, n),
            "HalfBath": rng.integers(0, 2, n),
            "BsmtFullBath": rng.integers(0, 2, n),
            "BsmtHalfBath": rng.integers(0, 2, n),
            "YearBuilt": rng.integers(1900, 2010, n),
            "YrSold": rng.integers(2006, 2011, n),
            "YearRemodAdd": rng.integers(1950, 2010, n),
            "PoolArea": np.where(rng.random(n) > 0.9, 400, 0),
            "GarageArea": rng.integers(0, 800, n),
            "OpenPorchSF": rng.integers(0, 300, n),
            "EnclosedPorch": rng.integers(0, 200, n),
            "3SsnPorch": np.zeros(n, dtype=int),
            "ScreenPorch": np.zeros(n, dtype=int),
            "Neighborhood": rng.choice(["NridgHt", "OldTown", "CollgCr"], n),
            "MSZoning": rng.choice(["RL", "RM", "FV"], n),
        }
    )
    y = 20000 * df["OverallQual"] + 100 * df["GrLivArea"] + rng.normal(0, 5000, n)
    return df, pd.Series(y, name="SalePrice")


@pytest.fixture
def toy_matrix(toy_frame):
    df, _ = toy_frame
    X = coerce_types(add_engineered_features(df))
    return X


@pytest.fixture
def models_dir(tmp_path, toy_matrix) -> str:
    """在临时目录里注册一个三件套模型（点预测 + 上下界）。"""
    X = toy_matrix
    point = build_pipeline(DummyRegressor(strategy="constant", constant=12.0), X)
    point.fit(X, np.full(len(X), 12.0))
    artifact = {
        "point": point,
        "interval": {"q_low": -0.4, "q_high": 0.4, "quantiles": [0.1, 0.9], "coverage_oof": 0.8},
    }

    metadata = {
        "columns": X.columns.tolist(),
        "defaults": default_values(X),
        "schema": schema(X),
        "engineered_features": ["TotalSF"],
    }
    metrics = {"mae": 18000.0, "rmse": 25000.0, "r2": 0.88, "mape": 9.5}
    registry.save_version(artifact, "DummyTest", metrics, metadata, models_dir=tmp_path)
    return str(tmp_path)
