"""注册表与指标计算。"""

from __future__ import annotations

import numpy as np
import pytest

from houseprice.features.build import build_matrix
from houseprice.models import registry
from houseprice.models.evaluate import regression_metrics


def test_regression_metrics_are_correct():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])
    result = regression_metrics(y_true, y_pred)
    assert result["mae"] == pytest.approx(6.67, abs=0.01)
    assert result["rmse"] == pytest.approx(np.sqrt(200 / 3), abs=0.01)
    # SS_res = 200，SS_tot = 20000 → R² = 1 - 200/20000 = 0.99
    assert result["r2"] == pytest.approx(0.99, abs=1e-4)


def test_registry_roundtrip(models_dir):
    artifact, metadata, version = registry.load_active(models_dir)
    assert version == "v1"
    assert set(artifact) == {"point", "interval"}
    assert artifact["interval"]["q_low"] < artifact["interval"]["q_high"]
    assert "OverallQual" in metadata["columns"]
    assert metadata["algorithm"] == "DummyTest"


def test_registry_activate_unknown_version(models_dir):
    with pytest.raises(KeyError):
        registry.activate("v99", models_dir)


def test_new_version_becomes_active(models_dir, toy_matrix):
    artifact, metadata, _ = registry.load_active(models_dir)
    new_version = registry.save_version(artifact, "DummyTest", {"mae": 1.0}, metadata, models_dir=models_dir)
    assert new_version == "v2"

    active, _, version = registry.load_active(models_dir)
    assert version == "v2"
    assert [item["version"] for item in registry.list_versions(models_dir)] == ["v1", "v2"]


def test_build_matrix_fills_defaults(models_dir):
    _, metadata, _ = registry.load_active(models_dir)
    frame = build_matrix(
        {"OverallQual": 8, "GrLivArea": 1800},
        metadata["columns"],
        metadata["defaults"],
    )
    assert len(frame) == 1
    assert list(frame.columns) == metadata["columns"]
    assert frame.iloc[0]["OverallQual"] == 8
    assert frame.iloc[0]["Neighborhood"] == metadata["defaults"]["Neighborhood"]
    assert frame.iloc[0]["TotalSF"] >= 1800


def test_build_matrix_rejects_missing_columns(models_dir):
    _, metadata, _ = registry.load_active(models_dir)
    with pytest.raises(KeyError):
        build_matrix({}, ["OverallQual", "不存在的列"], metadata["defaults"])
