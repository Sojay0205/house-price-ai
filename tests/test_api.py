"""HTTP 接口契约：状态码、区间单调性、字段校验。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from houseprice.api.main import create_app


@pytest.fixture
def client(models_dir):
    with TestClient(create_app(models_dir)) as test_client:
        yield test_client


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] == "v1"
    assert body["n_features"] > 10


def test_predict_returns_interval_containing_point(client):
    response = client.post(
        "/predict", json={"features": {"OverallQual": 7, "GrLivArea": 1710, "YearBuilt": 2003}}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["currency"] == "USD"
    assert body["interval"]["low"] <= body["prediction"] <= body["interval"]["high"]
    assert body["interval"]["confidence"] == 0.8


def test_predict_without_features_uses_defaults(client):
    response = client.post("/predict", json={"features": {}})
    assert response.status_code == 200
    assert response.json()["prediction"] > 0


def test_unknown_feature_is_rejected(client):
    response = client.post("/predict", json={"features": {"不存在的字段": 1}})
    assert response.status_code == 422
    assert "未知特征" in response.json()["detail"]


def test_non_numeric_value_is_rejected(client):
    response = client.post("/predict", json={"features": {"OverallQual": "很高"}})
    assert response.status_code == 422


def test_batch_prediction(client):
    payload = {"records": [{"OverallQual": 5}, {"OverallQual": 9}]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"records": []}).status_code == 422


def test_models_list_and_switch(client):
    body = client.get("/models").json()
    assert body["active"] == "v1"

    response = client.post("/models/v1/activate")
    assert response.status_code == 200
    assert response.json()["active"] == "v1"

    assert client.post("/models/v9/activate").status_code == 404


def test_explain_returns_contributions(client):
    response = client.post("/explain", json={"features": {"OverallQual": 8}})
    assert response.status_code == 200
    body = response.json()
    assert body["items"]
    assert {"feature", "contribution", "value"} == set(body["items"][0])
