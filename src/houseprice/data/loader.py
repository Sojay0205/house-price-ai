"""数据加载：优先读本地缓存，缺失时从 OpenML 拉取 Ames Housing。

选用 Ames Housing 而不是 scikit-learn 已移除的 Boston 数据集，原因：
1. 1460×80 的规模更接近真实业务数据，含类别特征与大量缺失值；
2. 数据异质性足够支撑特征工程与模型选择的叙事；
3. Boston 的 B 特征存在伦理争议，sklearn 1.2 起已彻底移除。
"""

from __future__ import annotations

import pandas as pd

from houseprice.config import MAX_MISSING_RATIO, OPENML_NAME, OPENML_VERSION, RAW_DIR, TARGET


def raw_path():
    return RAW_DIR / "ames.csv"


def fetch_ames(force: bool = False) -> pd.DataFrame:
    """返回原始 DataFrame（含目标列 SalePrice）。

    force=False 时命中本地缓存直接返回，保证离线可复现。
    """
    path = raw_path()
    if path.exists() and not force:
        return pd.read_csv(path)

    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(name=OPENML_NAME, version=OPENML_VERSION, as_frame=True, parser="auto")
    df = bunch.frame.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if TARGET not in df.columns:
        raise ValueError(f"数据集中未找到目标列 {TARGET}，实际列：{list(df.columns)[:10]}...")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def drop_sparse_columns(df: pd.DataFrame, threshold: float = MAX_MISSING_RATIO) -> pd.DataFrame:
    """剔除缺失率过高的列，其余列交给预处理管道填补。"""
    keep = [c for c in df.columns if df[c].isna().mean() <= threshold]
    return df[keep].copy()


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """拆出特征矩阵与目标向量（目标为美元原值）。"""
    df = drop_sparse_columns(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(float)
    return X, y


def column_kinds(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """按 dtype 划分数值列与类别列。"""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def default_values(X: pd.DataFrame) -> dict[str, object]:
    """为每个特征计算兜底填充值：数值取中位数，类别取众数。

    推理时用户只传部分字段，其余字段用训练集中位数/众数补齐，
    避免"少传一个字段就 422"的糟糕体验。
    """
    defaults: dict[str, object] = {}
    for col in X.columns:
        series = X[col]
        if pd.api.types.is_numeric_dtype(series):
            value = series.median()
            defaults[col] = 0.0 if pd.isna(value) else round(float(value), 4)
        else:
            mode = series.mode(dropna=True)
            defaults[col] = str(mode.iloc[0]) if not mode.empty else "None"
    return defaults


def schema(X: pd.DataFrame) -> dict[str, str]:
    """记录每列类型，供 API 校验请求字段。"""
    numeric_cols, _ = column_kinds(X)
    return {c: ("numeric" if c in numeric_cols else "categorical") for c in X.columns}
