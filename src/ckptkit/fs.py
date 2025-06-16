from __future__ import annotations

import json
import os
import random
import shutil
import string
from pathlib import Path
from typing import Iterable, List

from .manifest import MANIFEST_NAME, read_manifest


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_tree(path: Path) -> None:
    for root, _, files in os.walk(path):
        for name in files:
            full = Path(root) / name
            with open(full, "rb") as f:
                os.fsync(f.fileno())
        fsync_dir(Path(root))


def disk_free_bytes(path: Path) -> int:
    stat = shutil.disk_usage(path)
    return stat.free


def list_checkpoints(root: Path) -> List[Path]:
    checkpoints: List[Path] = []
    if not root.exists():
        return checkpoints
    for child in root.iterdir():
        if child.is_symlink():
            continue
        if child.is_dir() and (child / MANIFEST_NAME).exists():
            checkpoints.append(child)
    checkpoints.sort()
    return checkpoints


def read_step(checkpoint: Path) -> int:
    try:
        manifest = read_manifest(checkpoint / MANIFEST_NAME)
        return manifest.step
    except Exception:
        return -1


def _tmp_name(prefix: str) -> str:
    rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f".{prefix}.{rand}"


def update_latest_pointer(root: Path, target: Path) -> None:
    link = root / "latest"
    tmp_name = root / _tmp_name("latest")
    try:
        os.symlink(target.name, tmp_name)
        os.replace(tmp_name, link)
        return
    except OSError:
        if tmp_name.exists():
            tmp_name.unlink(missing_ok=True)
    payload = {"latest": str(target.resolve())}
    tmp_file = root / _tmp_name("latest_json")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_file, root / "latest.json")


def safe_remove_checkpoint(path: Path) -> None:
    if not path.exists():
        return
    # Use rmtree; caller ensures it's safe to delete (not currently writing).
    shutil.rmtree(path)
