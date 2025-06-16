from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from . import hashing
from .manifest import MANIFEST_NAME, Manifest, read_manifest


class Reason(str, enum.Enum):
    MANIFEST_MISSING = "manifest_missing"
    MANIFEST_SCHEMA = "manifest_schema_invalid"
    FILE_MISSING = "file_missing"
    SIZE_MISMATCH = "size_mismatch"
    HASH_MISMATCH = "hash_mismatch"
    ZERO_SIZED = "zero_sized_file"
    SPLIT_BRAIN = "split_brain_step_mismatch"


@dataclass
class Issue:
    reason: Reason
    detail: str
    path: Optional[str] = None


@dataclass
class ValidationResult:
    checkpoint: Path
    valid: bool
    issues: List[Issue] = field(default_factory=list)
    manifest: Optional[Manifest] = None

    def summary(self) -> str:
        if self.valid:
            return f"{self.checkpoint} valid"
        reasons = ", ".join(f"{i.reason.value}:{i.path or ''}" for i in self.issues)
        return f"{self.checkpoint} invalid [{reasons}]"


def validate_checkpoint(
    checkpoint: Path,
    *,
    full_hash: bool = False,
    sample_bytes: Optional[int] = 65536,
) -> ValidationResult:
    issues: List[Issue] = []
    manifest_path = checkpoint / MANIFEST_NAME
    if not manifest_path.exists():
        issues.append(Issue(Reason.MANIFEST_MISSING, "manifest missing"))
        return ValidationResult(checkpoint=checkpoint, valid=False, issues=issues)
    try:
        manifest = read_manifest(manifest_path)
    except Exception as exc:  # pragma: no cover - defensive
        issues.append(Issue(Reason.MANIFEST_SCHEMA, f"manifest load failed: {exc}"))
        return ValidationResult(checkpoint=checkpoint, valid=False, issues=issues)
    # Split brain detection based on directory naming convention step-<n> if present.
    m = re.search(r"(\d+)", checkpoint.name)
    if m:
        dir_step = int(m.group(1))
        if dir_step != manifest.step:
            issues.append(
                Issue(
                    Reason.SPLIT_BRAIN,
                    f"dir step {dir_step} differs from manifest {manifest.step}",
                    path=str(checkpoint),
                )
            )
    for entry in manifest.files:
        file_path = checkpoint / entry.path
        if not file_path.exists():
            issues.append(Issue(Reason.FILE_MISSING, "missing file", path=entry.path))
            continue
        size = file_path.stat().st_size
        if size == 0:
            issues.append(Issue(Reason.ZERO_SIZED, "zero-sized file", path=entry.path))
        if size != entry.size:
            issues.append(
                Issue(
                    Reason.SIZE_MISMATCH,
                    f"expected {entry.size} got {size}",
                    path=entry.path,
                )
            )
    if full_hash or sample_bytes is not None:
        # Hash only files that exist.
        existing_paths = [checkpoint / f.path for f in manifest.files if (checkpoint / f.path).exists()]
        hashes = hashing.hash_paths(existing_paths, sample_bytes=None if full_hash else sample_bytes)
        for entry in manifest.files:
            file_path = checkpoint / entry.path
            if not file_path.exists():
                continue
            digest = hashes[file_path]
            if digest != entry.sha256:
                issues.append(
                    Issue(
                        Reason.HASH_MISMATCH,
                        f"expected {entry.sha256} got {digest}",
                        path=entry.path,
                    )
                )
    valid = not issues
    return ValidationResult(checkpoint=checkpoint, valid=valid, issues=issues, manifest=manifest)
