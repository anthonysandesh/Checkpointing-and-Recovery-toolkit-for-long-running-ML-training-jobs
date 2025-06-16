from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Dict, Optional

import yaml


@dataclasses.dataclass
class HashingConfig:
    sample_bytes: Optional[int] = 65536
    threads: int = 4
    full: bool = False


@dataclasses.dataclass
class RetentionConfig:
    keep_last: int = 3
    keep_every: Optional[int] = None


@dataclasses.dataclass
class MetricsConfig:
    textfile: Optional[str] = None
    pushgateway: Optional[str] = None
    pushgateway_job: str = "ckptkit"
    labels: Dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Config:
    root: pathlib.Path
    hashing: HashingConfig = dataclasses.field(default_factory=HashingConfig)
    retention: RetentionConfig = dataclasses.field(default_factory=RetentionConfig)
    metrics: MetricsConfig = dataclasses.field(default_factory=MetricsConfig)
    job_id: str = "unknown"
    run_id: str = "unknown"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Config":
        root = pathlib.Path(data.get("root", "."))
        hashing = HashingConfig(**data.get("hashing", {}))
        retention = RetentionConfig(**data.get("retention", {}))
        metrics = MetricsConfig(**data.get("metrics", {}))
        job_id = data.get("job_id", "unknown")
        run_id = data.get("run_id", "unknown")
        return Config(
            root=root,
            hashing=hashing,
            retention=retention,
            metrics=metrics,
            job_id=job_id,
            run_id=run_id,
        )

    @staticmethod
    def load(path: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> "Config":
        data: Dict[str, Any] = {}
        if path:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        overrides = overrides or {}
        merged = _merge_dicts(data, overrides)
        return Config.from_dict(merged)


def _merge_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            result[key] = _merge_dicts(base[key], value)
        else:
            result[key] = value
    return result
