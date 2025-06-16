from __future__ import annotations

from pathlib import Path
from typing import Any


def load_checkpoint(path: Path) -> Any:
    try:
        import deepspeed  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError("DeepSpeed is not installed; cannot load checkpoint") from exc
    engine = deepspeed.checkpointing.CheckpointEngine()  # type: ignore[attr-defined]
    state = engine.load_checkpoint(str(path))
    return state
