from __future__ import annotations

import dataclasses
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import hashing

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = "1"


@dataclasses.dataclass
class FileEntry:
    path: str
    size: int
    sha256: str


@dataclasses.dataclass
class Manifest:
    version: str
    created_at: float
    job_id: str
    run_id: str
    step: int
    host: str
    world_size: int
    files: List[FileEntry]
    framework: Optional[str] = None
    precision: Optional[str] = None
    model_name: Optional[str] = None
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "step": self.step,
            "host": self.host,
            "world_size": self.world_size,
            "files": [dataclasses.asdict(f) for f in self.files],
            "framework": self.framework,
            "precision": self.precision,
            "model_name": self.model_name,
            "extra": self.extra,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Manifest":
        files = [FileEntry(**f) for f in data.get("files", [])]
        return Manifest(
            version=str(data["version"]),
            created_at=float(data["created_at"]),
            job_id=str(data["job_id"]),
            run_id=str(data["run_id"]),
            step=int(data["step"]),
            host=str(data["host"]),
            world_size=int(data["world_size"]),
            files=files,
            framework=data.get("framework"),
            precision=data.get("precision"),
            model_name=data.get("model_name"),
            extra=data.get("extra", {}),
        )


def manifest_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / MANIFEST_NAME


def write_manifest(path: Path, manifest: Manifest) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, sort_keys=True, indent=2)
        f.flush()
        os.fsync(f.fileno())


def read_manifest(path: Path) -> Manifest:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _validate_manifest_schema(data)
    return Manifest.from_dict(data)


def compute_manifest(
    checkpoint_dir: Path,
    job_id: str,
    run_id: str,
    step: int,
    world_size: int,
    *,
    framework: Optional[str] = None,
    precision: Optional[str] = None,
    model_name: Optional[str] = None,
    sample_bytes: Optional[int] = 65536,
    threads: int = 4,
    extra: Optional[Dict[str, Any]] = None,
    ignore: Optional[Iterable[str]] = None,
) -> Manifest:
    ignore_names = set(ignore or [])
    ignore_names.add(MANIFEST_NAME)
    files: List[Path] = []
    for root, _, filenames in os.walk(checkpoint_dir):
        for name in filenames:
            if name in ignore_names:
                continue
            path = Path(root) / name
            files.append(path)
    rel_files = [f.relative_to(checkpoint_dir) for f in files]
    sizes = {rel: f.stat().st_size for rel, f in zip(rel_files, files)}
    hashes = hashing.hash_paths(
        files,
        sample_bytes=sample_bytes,
        threads=threads,
    )
    entries = [
        FileEntry(path=str(rel), size=sizes[rel], sha256=hashes[path])
        for rel, path in zip(rel_files, files)
    ]
    entries.sort(key=lambda f: f.path)
    manifest_obj = Manifest(
        version=MANIFEST_VERSION,
        created_at=time.time(),
        job_id=job_id,
        run_id=run_id,
        step=step,
        host=socket.gethostname(),
        world_size=world_size,
        files=entries,
        framework=framework,
        precision=precision,
        model_name=model_name,
        extra=extra or {},
    )
    return manifest_obj


def _validate_manifest_schema(data: Dict[str, Any]) -> None:
    required = [
        "version",
        "created_at",
        "job_id",
        "run_id",
        "step",
        "host",
        "world_size",
        "files",
    ]
    for key in required:
        if key not in data:
            raise ValueError(f"manifest missing field {key}")
    if not isinstance(data["files"], list):
        raise ValueError("manifest files must be a list")
    for entry in data["files"]:
        for key in ("path", "size", "sha256"):
            if key not in entry:
                raise ValueError(f"manifest file missing {key}")
