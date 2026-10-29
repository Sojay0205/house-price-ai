"""预测区间：交叉验证残差的 conformal 校准。

为什么不用"直接用分位数回归模型"：在 1460 条样本上，LightGBM 分位数模型容易
欠拟合，实测覆盖率只有 54%（目标是 80%）——区间看起来窄，其实是模型没学到
条件分位点，等于给用户一个假的确定性。

改用 split-conformal 的思路：在训练集内做 K 折 OOF 预测，取残差的 10%/90% 分位
作为加性校准量。这样区间宽度由真实残差分布决定，覆盖率在留出集上接近名义水平，
而且只有两个标量参数，推理开销为零。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.base import clone

from houseprice.data.preprocess import build_pipeline


@dataclass
class ConformalInterval:
    q_low: float
    q_high: float
    quantiles: tuple[float, float]
    coverage_oof: float

    @classmethod
    def fit(cls, model, X, y_log: np.ndarray, splits, quantiles=(0.1, 0.9)) -> ConformalInterval:
        """用 OOF 残差的分位数做校准。"""
        oof = np.full(len(y_log), np.nan)
        for train_idx, valid_idx in splits:
            pipe = build_pipeline(clone(model), X.iloc[train_idx])
            pipe.fit(X.iloc[train_idx], y_log[train_idx])
            oof[valid_idx] = pipe.predict(X.iloc[valid_idx])

        residuals = y_log - oof
        q_low, q_high = np.quantile(residuals, quantiles)
        coverage = float(np.mean((residuals >= q_low) & (residuals <= q_high)))
        return cls(float(q_low), float(q_high), tuple(quantiles), coverage)

    def predict_log(self, point_log: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        point_log = np.asarray(point_log, dtype=float)
        return point_log + self.q_low, point_log + self.q_high

    def to_dict(self) -> dict:
        data = asdict(self)
        data["quantiles"] = list(self.quantiles)
        return data
