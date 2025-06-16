from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def load_checkpoint(path: Path, *, map_location: Optional[str] = None) -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError("PyTorch is not installed; cannot load checkpoint") from exc
    candidates = [
        path / "model.pt",
        path / "pytorch_model.bin",
        path / "model.bin",
    ]
    target = next((p for p in candidates if p.exists()), None)
    if target is None:
        raise FileNotFoundError(f"No known PyTorch checkpoint file in {path}")
    return torch.load(target, map_location=map_location)
