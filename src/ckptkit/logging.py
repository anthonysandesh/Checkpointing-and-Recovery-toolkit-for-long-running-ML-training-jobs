from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, Optional


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("ckptkit")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    return logger


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if isinstance(record.args, dict):
            payload.update(record.args)
        for key in ("run_id", "job_id", "step", "checkpoint_path", "event", "reason"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, sort_keys=True)


def log_event(
    logger: logging.Logger,
    *,
    event: str,
    severity: str = "INFO",
    reason: Optional[str] = None,
    **fields: Any,
) -> None:
    level = getattr(logging, severity.upper(), logging.INFO)
    extra: Dict[str, Any] = dict(fields)
    extra["event"] = event
    if reason:
        extra["reason"] = reason
    logger.log(level, event, extra=extra)
