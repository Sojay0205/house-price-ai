"""特征构造与单样本组装。

派生特征的思路是"把领域知识显式写进特征"，而不是指望模型自己发现交互项：
面积要合并地上地下、卫生间要按整卫/半卫折算、房龄要用成交年份减建造年份。
"""

from __future__ import annotations

import pandas as pd

from houseprice.data.preprocess import coerce_types


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """安全取数值列，缺失或不存在时返回默认值序列。"""
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """追加 9 个派生特征，保持原列不变。"""
    df = df.copy()

    df["TotalSF"] = _num(df, "TotalBsmtSF") + _num(df, "1stFlrSF") + _num(df, "2ndFlrSF")
    df["TotalBath"] = (
        _num(df, "FullBath")
        + 0.5 * _num(df, "HalfBath")
        + _num(df, "BsmtFullBath")
        + 0.5 * _num(df, "BsmtHalfBath")
    )
    df["TotalPorchSF"] = (
        _num(df, "OpenPorchSF")
        + _num(df, "EnclosedPorch")
        + _num(df, "3SsnPorch")
        + _num(df, "ScreenPorch")
    )
    df["HouseAge"] = _num(df, "YrSold", 2010) - _num(df, "YearBuilt", 1970)
    df["RemodAge"] = _num(df, "YrSold", 2010) - _num(df, "YearRemodAdd", 1970)
    df["HasPool"] = (_num(df, "PoolArea") > 0).astype(int)
    df["Has2ndFloor"] = (_num(df, "2ndFlrSF") > 0).astype(int)
    df["HasGarage"] = (_num(df, "GarageArea") > 0).astype(int)
    df["QualSF"] = _num(df, "OverallQual", 5) * _num(df, "GrLivArea")

    return df


def build_matrix(
    raw: dict[str, object],
    columns: list[str],
    defaults: dict[str, object],
) -> pd.DataFrame:
    """把一条原始请求组装成模型可接受的单行 DataFrame。

    缺失字段用训练集统计量兜底，字段顺序与训练时严格一致。
    """
    row: dict[str, object] = dict(defaults)
    for key, value in raw.items():
        if value is not None:
            row[key] = value

    df = pd.DataFrame([row])
    df = coerce_types(df)
    df = add_engineered_features(df)

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"缺少必要字段：{missing[:5]}")

    return df[columns]
