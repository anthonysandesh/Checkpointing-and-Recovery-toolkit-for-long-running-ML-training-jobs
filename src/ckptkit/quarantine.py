from __future__ import annotations

import datetime
import uuid
from pathlib import Path

from .fs import ensure_dir, fsync_dir


def quarantine(checkpoint: Path, *, root: Path, reason: str) -> Path:
    corrupt_dir = root / "corrupt"
    ensure_dir(corrupt_dir)
    target = corrupt_dir / f"{checkpoint.name}-{uuid.uuid4().hex}"
    checkpoint.replace(target)
    info = target / "reason.txt"
    with open(info, "w", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.utcnow().isoformat()}Z {reason}\n")
        f.flush()
    fsync_dir(corrupt_dir)
    return target
