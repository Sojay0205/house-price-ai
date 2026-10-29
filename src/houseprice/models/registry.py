"""模型注册表：版本化保存、加载与切换。

目录结构：
    models/
    ├── registry.json          # {"active": "v2", "versions": [...]}
    ├── v1/{model.joblib, metadata.json}
    └── v2/...
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from houseprice import config

REGISTRY_FILE = "registry.json"
MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


def _base(models_dir: str | Path | None = None) -> Path:
    return Path(models_dir) if models_dir is not None else Path(config.MODELS_DIR)


def _registry_path(models_dir: str | Path | None = None) -> Path:
    return _base(models_dir) / REGISTRY_FILE


def read_registry(models_dir: str | Path | None = None) -> dict:
    path = _registry_path(models_dir)
    if not path.exists():
        return {"active": None, "versions": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_registry(registry: dict, models_dir: str | Path | None = None) -> None:
    path = _registry_path(models_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def list_versions(models_dir: str | Path | None = None) -> list[dict]:
    return read_registry(models_dir)["versions"]


def next_version(models_dir: str | Path | None = None) -> str:
    base = _base(models_dir)
    used = {item["version"] for item in list_versions(models_dir)}
    index = len(used) + 1
    while f"v{index}" in used or (base / f"v{index}").exists():
        index += 1
    return f"v{index}"


def save_version(
    artifact: dict,
    algorithm: str,
    metrics: dict,
    metadata: dict,
    models_dir: str | Path | None = None,
    version: str | None = None,
    activate_now: bool = True,
) -> str:
    """保存一版模型并登记到注册表，默认设为线上版本。"""
    base = _base(models_dir)
    version = version or next_version(models_dir)
    target = base / version
    target.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, target / MODEL_FILE, compress=3)
    meta = {
        "version": version,
        "algorithm": algorithm,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "metrics": metrics,
        **metadata,
    }
    (target / METADATA_FILE).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    registry = read_registry(models_dir)
    registry["versions"] = [
        item for item in registry["versions"] if item["version"] != version
    ] + [
        {
            "version": version,
            "algorithm": algorithm,
            "created_at": meta["created_at"],
            "metrics": metrics,
        }
    ]
    registry["versions"].sort(key=lambda item: int(item["version"].lstrip("v")))
    if activate_now or not registry.get("active"):
        registry["active"] = version
    _write_registry(registry, models_dir)
    return version


def load_version(
    version: str, models_dir: str | Path | None = None
) -> tuple[dict, dict]:
    target = _base(models_dir) / version
    model_path = target / MODEL_FILE
    meta_path = target / METADATA_FILE
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"模型版本 {version} 不存在：{target}")
    artifact = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return artifact, metadata


def load_active(models_dir: str | Path | None = None) -> tuple[dict, dict, str]:
    registry = read_registry(models_dir)
    version = registry.get("active")
    if not version:
        raise FileNotFoundError("注册表中没有可用模型，请先执行 `make train`")
    artifact, metadata = load_version(version, models_dir)
    return artifact, metadata, version


def activate(version: str, models_dir: str | Path | None = None) -> dict:
    """切换线上版本（热切换由服务端下次加载时生效）。"""
    available = {item["version"] for item in list_versions(models_dir)}
    if version not in available:
        raise KeyError(f"版本 {version} 未注册，现有版本：{sorted(available)}")
    registry = read_registry(models_dir)
    registry["active"] = version
    _write_registry(registry, models_dir)
    return registry
