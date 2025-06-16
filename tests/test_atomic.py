import json
import pytest
from pathlib import Path

from ckptkit.atomic import atomic_checkpoint_write
from ckptkit.config import RetentionConfig
from ckptkit.manifest import compute_manifest, manifest_path, write_manifest


def test_atomic_write_crash_cleans_tmp(tmp_path: Path) -> None:
    dest = tmp_path / "step-1"

    def writer(temp: Path):
        (temp / "file.bin").write_bytes(b"hello")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        atomic_checkpoint_write(dest, writer, update_latest=False, retention=None)

    assert not dest.exists()


def test_atomic_write_retention_and_latest(tmp_path: Path) -> None:
    retention = RetentionConfig(keep_last=1)

    def make_writer(step: int):
        def writer(temp: Path):
            (temp / "state.json").write_text(json.dumps({"step": step}), encoding="utf-8")
            manifest = compute_manifest(temp, job_id="job", run_id="run", step=step, world_size=1)
            write_manifest(manifest_path(temp), manifest)
            return manifest

        return writer

    ckpt1 = tmp_path / "step-1"
    atomic_checkpoint_write(ckpt1, make_writer(1), retention=retention)
    ckpt2 = tmp_path / "step-2"
    atomic_checkpoint_write(ckpt2, make_writer(2), retention=retention)

    assert not ckpt1.exists()  # removed by retention
    assert ckpt2.exists()
    latest_link = tmp_path / "latest"
    latest_json = tmp_path / "latest.json"
    assert latest_link.exists() or latest_json.exists()
