from __future__ import annotations

import datetime
import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

from .fs import disk_free_bytes
from .validate import ValidationResult, Reason
from .resume import ResumePlan

LabelMap = Mapping[str, str]


@dataclass
class MetricSample:
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"

    def render(self) -> str:
        label_str = ""
        if self.labels:
            joined = ",".join(f'{k}="{v}"' for k, v in sorted(self.labels.items()))
            label_str = f"{{{joined}}}"
        return f"{self.name}{label_str} {self.value}"


class MetricsEmitter:
    def __init__(self, base_labels: LabelMap | None = None):
        self.samples: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], MetricSample] = {}
        self.base_labels = dict(base_labels or {})

    def _key(self, name: str, labels: LabelMap) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
        merged = {**self.base_labels, **labels}
        return name, tuple(sorted(merged.items()))

    def gauge(self, name: str, value: float, labels: LabelMap | None = None) -> None:
        labels = labels or {}
        key = self._key(name, labels)
        merged_labels = {**self.base_labels, **labels}
        self.samples[key] = MetricSample(name=name, value=value, labels=merged_labels, metric_type="gauge")

    def counter(self, name: str, value: float, labels: LabelMap | None = None) -> None:
        labels = labels or {}
        key = self._key(name, labels)
        merged_labels = {**self.base_labels, **labels}
        if key in self.samples:
            self.samples[key].value += value
        else:
            self.samples[key] = MetricSample(name=name, value=value, labels=merged_labels, metric_type="counter")

    def text(self) -> str:
        lines = []
        for sample in sorted(self.samples.values(), key=lambda s: (s.name, sorted(s.labels.items()))):
            lines.append(sample.render())
        return "\n".join(lines) + "\n"

    def write_textfile(self, path: Path) -> None:
        tmp = Path(tempfile.mkstemp(prefix=".ckptkit", dir=path.parent)[1])
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(self.text())
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def push_gateway(self, url: str, job: str = "ckptkit") -> None:
        target = url.rstrip("/") + f"/metrics/job/{job}"
        data = self.text().encode("utf-8")
        req = urllib.request.Request(target, data=data, method="PUT")
        req.add_header("Content-Type", "text/plain")
        try:
            urllib.request.urlopen(req, timeout=5)  # nosec B310
        except urllib.error.URLError as exc:
            raise RuntimeError(f"pushgateway push failed: {exc}") from exc


def record_validation_metrics(emitter: MetricsEmitter, results: Iterable[ValidationResult]) -> None:
    failure_reasons: Dict[str, int] = {}
    total_failures = 0
    for res in results:
        if res.valid:
            continue
        total_failures += 1
        for issue in res.issues:
            failure_reasons[issue.reason.value] = failure_reasons.get(issue.reason.value, 0) + 1
    for reason, count in failure_reasons.items():
        emitter.counter("checkpoint_validation_failures_total", count, labels={"reason": reason})
    emitter.counter("checkpoint_validation_failures_total", total_failures, labels={"reason": "all"})


def record_resume_plan(emitter: MetricsEmitter, plan: ResumePlan) -> None:
    emitter.gauge("checkpoint_resume_selected_step", float(plan.step))
    emitter.gauge(
        "checkpoint_corrupt_detected",
        0.0 if plan.validation.valid else 1.0,
        labels={"checkpoint": plan.checkpoint.name},
    )


def record_checkpoint_write(
    emitter: MetricsEmitter,
    *,
    checkpoint_path: Path,
    manifest_step: int,
    duration_seconds: float,
    total_bytes: float,
) -> None:
    emitter.gauge("checkpoint_last_success_step", float(manifest_step))
    emitter.gauge("checkpoint_last_success_timestamp", datetime.datetime.utcnow().timestamp())
    emitter.gauge("checkpoint_last_duration_seconds", duration_seconds)
    emitter.counter("checkpoint_write_bytes_total", total_bytes)
    # Emit histogram-style buckets for duration.
    buckets = [0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1200, 3600]
    for b in buckets:
        emitter.counter(
            "checkpoint_last_duration_seconds_bucket",
            1.0 if duration_seconds <= b else 0.0,
            labels={"le": str(b)},
        )
    emitter.counter("checkpoint_last_duration_seconds_bucket", 1.0, labels={"le": "+Inf"})
    emitter.counter("checkpoint_last_duration_seconds_count", 1.0)
    emitter.counter("checkpoint_last_duration_seconds_sum", duration_seconds)


def record_disk_free(emitter: MetricsEmitter, path: Path) -> None:
    emitter.gauge("checkpoint_directory_free_bytes", float(disk_free_bytes(path)))
