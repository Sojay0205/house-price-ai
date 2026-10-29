"""预处理管道：形状、缺失值、未知类别三项必须成立。"""

from __future__ import annotations

import numpy as np

from houseprice.data.preprocess import build_pipeline, coerce_types
from houseprice.features.build import add_engineered_features


def test_pipeline_output_is_finite(toy_matrix):
    pipe = build_pipeline(None, toy_matrix)
    matrix = pipe.named_steps["preprocess"].fit_transform(toy_matrix)
    assert matrix.shape[0] == len(toy_matrix)
    assert np.isfinite(matrix).all(), "预处理后不应存在 NaN 或 inf"


def test_unseen_category_is_ignored(toy_matrix):
    preprocess = build_pipeline(None, toy_matrix).named_steps["preprocess"]
    preprocess.fit(toy_matrix)

    unseen = toy_matrix.head(1).copy()
    unseen["Neighborhood"] = "从未出现过的社区"
    matrix = preprocess.transform(unseen)
    assert matrix.shape[1] == preprocess.transform(toy_matrix.head(1)).shape[1]


def test_engineered_features_created(toy_frame):
    df, _ = toy_frame
    X = coerce_types(add_engineered_features(df))
    for column in ("TotalSF", "TotalBath", "HouseAge", "QualSF", "HasGarage"):
        assert column in X.columns

    row = X.iloc[0]
    assert row["TotalSF"] == row["TotalBsmtSF"] + row["1stFlrSF"] + row["2ndFlrSF"]
    assert row["QualSF"] == row["OverallQual"] * row["GrLivArea"]


def test_missing_values_are_imputed(toy_matrix):
    sparse = toy_matrix.copy()
    sparse.loc[sparse.index[:20], "GrLivArea"] = np.nan
    preprocess = build_pipeline(None, sparse).named_steps["preprocess"]
    matrix = preprocess.fit_transform(sparse)
    assert np.isfinite(matrix).all()
