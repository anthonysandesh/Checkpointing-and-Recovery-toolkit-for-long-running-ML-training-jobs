from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Optional, Set

from .config import RetentionConfig
from .fs import ensure_dir, fsync_dir, fsync_tree, list_checkpoints, read_step, safe_remove_checkpoint, update_latest_pointer
from .manifest import MANIFEST_NAME, Manifest, manifest_path, write_manifest


def atomic_rename(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    src_parent = src.parent
    dst_parent = dst.parent
    fsync_dir(src_parent)
    fsync_dir(dst_parent)
    src.replace(dst)
    fsync_dir(dst_parent)


def atomic_checkpoint_write(
    dest_dir: Path,
    write_fn: Callable[[Path], Manifest],
    *,
    update_latest: bool = True,
    retention: Optional[RetentionConfig] = None,
) -> Manifest:
    parent = dest_dir.parent
    ensure_dir(parent)
    temp_dir_path = Path(tempfile.mkdtemp(prefix=dest_dir.name + ".tmp-", dir=parent))
    manifest: Manifest
    try:
        manifest = write_fn(temp_dir_path)
        if not (temp_dir_path / MANIFEST_NAME).exists():
            # Ensure manifest is written by the writer.
            write_manifest(manifest_path(temp_dir_path), manifest)
        fsync_tree(temp_dir_path)
        fsync_dir(parent)
        atomic_rename(temp_dir_path, dest_dir)
    finally:
        # If rename failed, ensure temp dir is cleaned up.
        if temp_dir_path.exists() and temp_dir_path != dest_dir:
            shutil.rmtree(temp_dir_path, ignore_errors=True)
    if update_latest:
        update_latest_pointer(parent, dest_dir)
    if retention:
        apply_retention(parent, retention, keep_paths={dest_dir})
    return manifest


def apply_retention(root: Path, retention: RetentionConfig, *, keep_paths: Optional[Iterable[Path]] = None) -> None:
    keep_set: Set[Path] = set(keep_paths or [])
    checkpoints = list_checkpoints(root)
    checkpoints.sort(key=read_step)
    survivors: Set[Path] = set()
    if retention.keep_last:
        survivors.update(checkpoints[-retention.keep_last :])
    if retention.keep_every:
        for ckpt in checkpoints:
            step = read_step(ckpt)
            if step >= 0 and step % retention.keep_every == 0:
                survivors.add(ckpt)
    survivors.update(keep_set)
    for ckpt in checkpoints:
        if ckpt in survivors:
            continue
        safe_remove_checkpoint(ckpt)
