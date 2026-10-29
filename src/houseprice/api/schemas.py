"""API 数据契约（Pydantic v2）。

请求体只要求传关心的字段，缺失字段由服务端用训练集统计量兜底——
房子有 80 个属性，强迫调用方每次都传全量字段是不现实的。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Interval(BaseModel):
    low: float = Field(..., description="区间下界（美元）")
    high: float = Field(..., description="区间上界（美元）")
    confidence: float = Field(0.8, description="区间置信水平")


class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="点预测房价（美元）")
    interval: Interval
    currency: str = "USD"
    model_version: str
    algorithm: str


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any] = Field(
        default_factory=dict,
        description="原始特征名 -> 取值，例如 {'OverallQual': 7, 'GrLivArea': 1710}",
    )


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[dict[str, Any]] = Field(..., min_length=1, max_length=1000)


class BatchPredictResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class ModelVersion(BaseModel):
    version: str
    algorithm: str
    created_at: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class ModelsResponse(BaseModel):
    active: str
    versions: list[ModelVersion]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    algorithm: str
    n_features: int


class ContributionItem(BaseModel):
    feature: str
    contribution: float
    value: float


class ExplainResponse(BaseModel):
    model_version: str
    base_value_log: float
    items: list[ContributionItem]
