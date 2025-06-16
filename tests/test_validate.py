from pathlib import Path

from ckptkit.manifest import compute_manifest, manifest_path, write_manifest
from ckptkit.validate import Reason, validate_checkpoint


def _make_checkpoint(tmp_path: Path, step: int) -> Path:
    ckpt = tmp_path / f"step-{step}"
    ckpt.mkdir()
    data = ckpt / "weights.bin"
    data.write_bytes(b"abc")
    manifest = compute_manifest(ckpt, job_id="job", run_id="run", step=step, world_size=1)
    write_manifest(manifest_path(ckpt), manifest)
    return ckpt


def test_checksum_mismatch_detected(tmp_path: Path) -> None:
    ckpt = _make_checkpoint(tmp_path, 1)
    # Corrupt file
    (ckpt / "weights.bin").write_bytes(b"bad")
    res = validate_checkpoint(ckpt, full_hash=True)
    assert not res.valid
    reasons = {issue.reason for issue in res.issues}
    assert Reason.HASH_MISMATCH in reasons


def test_missing_file_detected(tmp_path: Path) -> None:
    ckpt = _make_checkpoint(tmp_path, 2)
    (ckpt / "weights.bin").unlink()
    res = validate_checkpoint(ckpt, full_hash=False)
    assert not res.valid
    reasons = {issue.reason for issue in res.issues}
    assert Reason.FILE_MISSING in reasons
