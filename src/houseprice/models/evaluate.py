"""评估指标与诊断图。

模型在 log1p 空间训练（压制右偏与异方差），但对外汇报的指标换算回美元，
因为业务方关心的是"平均差多少钱"，而不是对数误差。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """美元口径的回归指标。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mape": round(mape, 2),
    }


def plot_pred_vs_actual(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str = "Predicted vs actual"
) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=140)
    ax.scatter(y_true, y_pred, s=14, alpha=0.55, edgecolor="white", linewidth=0.4, color="#378ADD")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "--", color="#A32D2D", linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel("Actual SalePrice (USD)")
    ax.set_ylabel("Predicted SalePrice (USD)")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> Path:
    residuals = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=140)
    ax.hist(residuals, bins=40, color="#5DCAA5", edgecolor="white")
    ax.axvline(0, color="#A32D2D", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Residual (USD, predicted − actual)")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution", fontsize=12)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
