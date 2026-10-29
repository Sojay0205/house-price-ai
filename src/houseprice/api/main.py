"""服务层：FastAPI 推理服务。

启动时从注册表加载 active 版本，模型常驻内存；版本切换后立即重载，
不需要重启容器。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException

from houseprice import __version__
from houseprice.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ExplainResponse,
    HealthResponse,
    Interval,
    ModelsResponse,
    PredictionResponse,
    PredictRequest,
)
from houseprice.explain.shap_explainer import explain_single
from houseprice.features.build import build_matrix
from houseprice.models import registry


def _validate_features(raw: dict[str, Any], known: dict[str, str]) -> dict[str, Any]:
    """拒绝未知字段，并把数值字段强制转成 float。"""
    unknown = [key for key in raw if key not in known]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"未知特征：{unknown[:5]}（共 {len(unknown)} 个）。字段名需与 Ames 原始列名一致。",
        )

    cleaned: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None:
            continue
        if known[key] == "numeric":
            try:
                cleaned[key] = float(value)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=422, detail=f"字段 {key} 需要数值，收到 {value!r}"
                ) from exc
        else:
            cleaned[key] = str(value)
    return cleaned


def create_app(models_dir: str | Path | None = None) -> FastAPI:
    """应用工厂，便于测试时注入临时模型目录。"""
    state: dict[str, Any] = {}

    def load(version: str | None = None) -> None:
        if version is None:
            artifact, metadata, active = registry.load_active(models_dir)
        else:
            artifact, metadata = registry.load_version(version, models_dir)
            active = version
        state.update(
            artifact=artifact,
            metadata=metadata,
            version=active,
            columns=metadata["columns"],
            defaults=metadata["defaults"],
            schema=metadata["schema"],
            algorithm=metadata["algorithm"],
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            load()
        except FileNotFoundError as exc:
            raise RuntimeError(
                "未找到可用模型。请先执行 `make train` 生成 models/v1/。"
            ) from exc
        yield
        state.clear()

    app = FastAPI(
        title="HousePrice-AI",
        description="房价预估服务：点预测 + 80% 预测区间 + SHAP 归因",
        version=__version__,
        lifespan=lifespan,
    )

    def predict_one(raw: dict[str, Any]) -> PredictionResponse:
        cleaned = _validate_features(raw, state["schema"])
        frame = build_matrix(cleaned, state["columns"], state["defaults"])

        point_log = float(state["artifact"]["point"].predict(frame)[0])
        calibration = state["artifact"]["interval"]
        point = max(float(np.expm1(point_log)), 0.0)
        low = max(float(np.expm1(point_log + calibration["q_low"])), 0.0)
        high = max(float(np.expm1(point_log + calibration["q_high"])), 0.0)
        low, high = min(low, point), max(high, point)
        return PredictionResponse(
            prediction=round(point, 2),
            interval=Interval(low=round(low, 2), high=round(high, 2)),
            model_version=state["version"],
            algorithm=state["algorithm"],
        )

    @app.get("/health", response_model=HealthResponse, summary="健康检查")
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_version=state["version"],
            algorithm=state["algorithm"],
            n_features=len(state["columns"]),
        )

    @app.post("/predict", response_model=PredictionResponse, summary="单条预测")
    def predict(payload: PredictRequest) -> PredictionResponse:
        return predict_one(payload.features)

    @app.post("/predict/batch", response_model=BatchPredictResponse, summary="批量预测")
    def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
        results = [predict_one(record) for record in payload.records]
        return BatchPredictResponse(count=len(results), predictions=results)

    @app.post("/explain", response_model=ExplainResponse, summary="单条预测的 SHAP 归因")
    def explain(payload: PredictRequest) -> ExplainResponse:
        cleaned = _validate_features(payload.features, state["schema"])
        frame = build_matrix(cleaned, state["columns"], state["defaults"])
        result = explain_single(state["artifact"]["point"], frame)
        return ExplainResponse(model_version=state["version"], **result)

    @app.get("/models", response_model=ModelsResponse, summary="模型版本列表")
    def models() -> ModelsResponse:
        return ModelsResponse(
            active=state["version"], versions=registry.list_versions(models_dir)
        )

    @app.post("/models/{version}/activate", response_model=ModelsResponse, summary="切换线上版本")
    def switch(version: str) -> ModelsResponse:
        try:
            registry.activate(version, models_dir)
            load(version)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return ModelsResponse(
            active=state["version"], versions=registry.list_versions(models_dir)
        )

    return app


app = create_app()
