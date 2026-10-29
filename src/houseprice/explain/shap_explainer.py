"""SHAP 可解释性。

全局图回答"哪些特征整体上最影响房价"，单样本分解回答"这一套房子为什么估这个价"。
注意：SHAP 作用在 Pipeline 预处理之后的矩阵上，因此特征名取自 ColumnTransformer
的 get_feature_names_out，而不是原始列名。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def _transform(pipe, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    preprocess = pipe.named_steps["preprocess"]
    matrix = preprocess.transform(X)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    names = list(preprocess.get_feature_names_out())
    return np.asarray(matrix), names


def _make_explainer(pipe, background: pd.DataFrame):
    matrix, names = _transform(pipe, background)
    model = pipe.named_steps["model"]
    try:
        return shap.TreeExplainer(model), matrix, names
    except Exception:
        # 非树模型（Ridge / MLP）退化到通用解释器，样本量控制在 100 以内保证速度
        sample = matrix[: min(100, len(matrix))]
        return shap.Explainer(model.predict, sample), matrix, names


def save_global_summary(pipe, X: pd.DataFrame, out_path: Path, max_display: int = 15) -> Path:
    """SHAP summary plot：按平均绝对贡献排序的蜂群图。"""
    explainer, matrix, names = _make_explainer(pipe, X)
    sample = matrix[: min(400, len(matrix))]
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values)

    fig = plt.figure(figsize=(8.0, 6.4), dpi=140)
    shap.summary_plot(values, sample, feature_names=names, max_display=max_display, show=False)
    plt.title("SHAP feature contribution (global)", fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def explain_single(pipe, row: pd.DataFrame, top_k: int = 10) -> dict:
    """单条样本的贡献度分解，返回 base_value 与 Top-K 特征。"""
    explainer, _, names = _make_explainer(pipe, row)
    matrix, _ = _transform(pipe, row)
    values = explainer.shap_values(matrix)
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values).reshape(-1)

    order = np.argsort(np.abs(values))[::-1][:top_k]
    base_value = float(np.ravel(getattr(explainer, "expected_value", 0.0))[0])
    items = [
        {
            "feature": names[i],
            "contribution": round(float(values[i]), 4),
            "value": round(float(matrix[0, i]), 4),
        }
        for i in order
    ]
    return {"base_value_log": round(base_value, 4), "items": items}
