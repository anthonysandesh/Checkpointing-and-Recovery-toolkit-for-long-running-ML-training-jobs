from pathlib import Path

from ckptkit.manifest import compute_manifest, manifest_path, write_manifest
from ckptkit.resume import Policy, select_checkpoint


def _write_checkpoint(path: Path, step: int, corrupt: bool = False) -> None:
    path.mkdir()
    data = path / "weights.bin"
    data.write_bytes(b"good" if not corrupt else b"bad")
    manifest = compute_manifest(path, job_id="job", run_id="run", step=step, world_size=1)
    if corrupt:
        # Point manifest to missing file to force failure
        manifest.files[0].sha256 = "deadbeef"
    write_manifest(manifest_path(path), manifest)


def test_resume_fallback_to_last_valid(tmp_path: Path) -> None:
    root = tmp_path
    ckpt1 = root / "step-1"
    ckpt2 = root / "step-2"
    _write_checkpoint(ckpt1, step=1, corrupt=False)
    _write_checkpoint(ckpt2, step=2, corrupt=True)

    plan = select_checkpoint(root, policy=Policy.LATEST_VALID)
    assert plan.step == 1
    assert plan.validation.valid
