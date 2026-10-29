"""预处理管道。

关键设计：数值与类别分支封装进 ColumnTransformer，并与模型一起组成
sklearn Pipeline。Pipeline 序列化后同时携带"填补中位数、缩放参数、独热
编码词表"，推理端不需要任何手工复现，从根上消除 train/serve skew。
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from houseprice.data.loader import column_kinds


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """统一 dtype：数值列转 float，类别列转 str。

    OpenML 下发的类别列是 category dtype，直接用 str 填补会报错；
    统一转 object 后再交给 SimpleImputer。
    """
    df = df.copy()
    numeric_cols, categorical_cols = column_kinds(df)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in categorical_cols:
        df[col] = df[col].astype(object).where(df[col].notna(), "None").astype(str)
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """按列类型构建预处理分支。列清单只依赖列名与 dtype，不触碰具体取值，因此不产生泄漏。"""
    numeric_cols, categorical_cols = column_kinds(X)

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(model, X: pd.DataFrame) -> Pipeline:
    """预处理 + 模型，作为一个整体训练与保存。"""
    return Pipeline([("preprocess", build_preprocessor(X)), ("model", model)])
