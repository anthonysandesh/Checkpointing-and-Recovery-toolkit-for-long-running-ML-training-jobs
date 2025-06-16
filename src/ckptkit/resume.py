from __future__ import annotations

import json
import enum
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .fs import list_checkpoints, read_step, update_latest_pointer
from .validate import ValidationResult, validate_checkpoint


class Policy(str, enum.Enum):
    LATEST_VALID = "latest-valid"
    LAST_KNOWN_GOOD = "last-known-good"
    NEWEST_BEFORE = "newest-before"
    BEST = "best"


@dataclass
class ResumePlan:
    checkpoint: Path
    step: int
    reason: str
    validation: ValidationResult


def _latest_pointer(root: Path) -> Optional[Path]:
    link = root / "latest"
    json_path = root / "latest.json"
    if link.exists() or link.is_symlink():
        try:
            target = link.resolve(strict=False)
            if target.exists():
                return target
        except OSError:
            return None
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            target = Path(payload.get("latest", ""))
            if target.exists():
                return target
        except Exception:
            return None
    return None


def _validate_candidates(candidates: List[Path], *, full_hash: bool = False) -> List[ValidationResult]:
    results: List[ValidationResult] = []
    for ckpt in candidates:
        results.append(validate_checkpoint(ckpt, full_hash=full_hash))
    return results


def select_checkpoint(
    root: Path,
    policy: Policy = Policy.LATEST_VALID,
    *,
    before_step: Optional[int] = None,
    full_hash: bool = False,
    repair_latest: bool = True,
) -> ResumePlan:
    candidates = list_checkpoints(root)
    validations = _validate_candidates(candidates, full_hash=full_hash)
    validations.sort(key=lambda r: read_step(r.checkpoint), reverse=True)
    latest_path = _latest_pointer(root)

    def pick_first(predicate) -> Optional[ValidationResult]:
        for v in validations:
            if predicate(v):
                return v
        return None

    chosen: Optional[ValidationResult] = None
    reason = ""
    if policy == Policy.LATEST_VALID:
        chosen = pick_first(lambda v: v.valid)
        reason = "latest valid checkpoint"
    elif policy == Policy.LAST_KNOWN_GOOD:
        if latest_path:
            for v in validations:
                if v.checkpoint == latest_path and v.valid:
                    chosen = v
                    reason = "latest pointer valid"
                    break
        if chosen is None:
            chosen = pick_first(lambda v: v.valid)
            reason = "fallback to newest valid"
    elif policy == Policy.NEWEST_BEFORE:
        if before_step is None:
            raise ValueError("before_step required for newest-before policy")
        chosen = pick_first(lambda v: v.valid and v.manifest and v.manifest.step <= before_step)
        reason = f"newest valid checkpoint before {before_step}"
    elif policy == Policy.BEST:
        chosen = pick_first(lambda v: v.valid)
        if chosen:
            reason = "best valid checkpoint"
        else:
            chosen = validations[0] if validations else None
            reason = "no valid checkpoints; using newest even if invalid"

    if not chosen:
        raise RuntimeError(f"No checkpoints available under {root}")

    if repair_latest and chosen.valid:
        try:
            update_latest_pointer(root, chosen.checkpoint)
        except Exception:
            pass

    step = chosen.manifest.step if chosen.manifest else -1
    return ResumePlan(checkpoint=chosen.checkpoint, step=step, reason=reason, validation=chosen)
