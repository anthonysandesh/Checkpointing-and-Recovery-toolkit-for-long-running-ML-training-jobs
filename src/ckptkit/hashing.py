from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, Optional


def compute_sha256(path: Path, *, sample_bytes: Optional[int] = None, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    size = path.stat().st_size
    if sample_bytes is None or sample_bytes <= 0 or sample_bytes * 2 >= size:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    with open(path, "rb") as f:
        head = f.read(sample_bytes)
        h.update(head)
        if size > sample_bytes:
            # Seek to tail chunk for better detection of truncated writes.
            f.seek(max(size - sample_bytes, sample_bytes))
            tail = f.read(sample_bytes)
            h.update(tail)
    h.update(str(size).encode("utf-8"))
    return h.hexdigest()


def hash_paths(
    paths: Iterable[Path],
    *,
    sample_bytes: Optional[int] = None,
    threads: int = 4,
) -> Dict[Path, str]:
    results: Dict[Path, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, threads)) as executor:
        futures = {executor.submit(compute_sha256, path, sample_bytes=sample_bytes): path for path in paths}
        for fut in as_completed(futures):
            path = futures[fut]
            results[path] = fut.result()
    return results
