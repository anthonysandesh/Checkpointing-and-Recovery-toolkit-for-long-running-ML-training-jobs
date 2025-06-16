from pathlib import Path

from ckptkit.manifest import compute_manifest, manifest_path, read_manifest, write_manifest


def test_manifest_roundtrip(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    data = ckpt / "tensor.bin"
    data.write_bytes(b"\x01\x02")

    manifest = compute_manifest(
        ckpt,
        job_id="job",
        run_id="run",
        step=1,
        world_size=1,
        framework="pytorch",
        precision="fp32",
        model_name="toy",
    )
    write_manifest(manifest_path(ckpt), manifest)
    loaded = read_manifest(manifest_path(ckpt))

    assert loaded.step == manifest.step
    assert loaded.job_id == "job"
    assert loaded.files[0].path == "tensor.bin"
    assert loaded.files[0].size == 2
