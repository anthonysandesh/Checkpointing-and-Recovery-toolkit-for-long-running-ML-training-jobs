from pathlib import Path

from ckptkit.metrics import MetricsEmitter, record_validation_metrics
from ckptkit.validate import Issue, Reason, ValidationResult


def test_textfile_metrics_formatting(tmp_path: Path) -> None:
    emitter = MetricsEmitter({"job_id": "job"})
    invalid = ValidationResult(
        checkpoint=tmp_path / "ckpt",
        valid=False,
        issues=[Issue(reason=Reason.HASH_MISMATCH, detail="bad", path="x")],
        manifest=None,
    )
    record_validation_metrics(emitter, [invalid])
    emitter.gauge("checkpoint_resume_selected_step", 5)
    prom = emitter.text()
    assert 'checkpoint_validation_failures_total{job_id="job",reason="hash_mismatch"} 1' in prom
    assert "checkpoint_resume_selected_step" in prom
    out_path = tmp_path / "metrics.prom"
    emitter.write_textfile(out_path)
    assert out_path.read_text()
