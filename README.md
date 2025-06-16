# Checkpointing & Recovery Toolkit (ckptkit)

Production-grade checkpoint writer, validator, and recovery utilities for long-running ML training jobs (PyTorch, DeepSpeed, etc.). Provides atomic checkpoint creation with manifests, corruption detection, resume planning, Prometheus metrics, and ready-to-import Grafana dashboards.

## Quickstart

```bash
pip install -e .
ckptkit write --root /tmp/ckpts --job-id demo --run-id run1 --step 10
ckptkit validate /tmp/ckpts/step-10
ckptkit scan /tmp/ckpts
ckptkit resume /tmp/ckpts
ckptkit emit-metrics --root /tmp/ckpts --textfile /tmp/ckpt_metrics.prom
```

Run periodic validation:
```bash
scripts/validate_loop.sh /tmp/ckpts 60
```

Watch for new corrupt checkpoints:
```bash
scripts/watch_corruption.sh /tmp/ckpts
```

Reference integration example:
```bash
python examples/toy_train_with_ckptkit.py --root /tmp/ckpts --steps 5
```

## Features
- Atomic checkpoint writes: temp dir + fsync + atomic rename, manifest with sha256 hashes
- Integrity validation (structure, sizes, hashes) with corruption detection and quarantine
- Resume planning (latest-valid, last-known-good, newest-before) with repair fallback
- Retention policy (keep last N, keep every K steps) and safe garbage collection
- Prometheus metrics (textfile + Pushgateway) and JSON logging helpers
- Grafana dashboard and Prometheus scrape config examples
- Linux scripts for periodic validation and corruption watching

## Configuration
YAML config with CLI overrides:
```yaml
root: /tmp/ckpts
hashing:
  sample_bytes: 65536
retention:
  keep_last: 3
  keep_every: 1000
metrics:
  textfile: /var/lib/node_exporter/ckptkit.prom
  pushgateway: http://pushgateway:9091/metrics
```

Load config with `--config path.yaml` or rely on CLI flags.

## CLI Commands
- `ckptkit write`: demo writer that produces a checkpoint with manifest and updates `latest`
- `ckptkit validate <path>`: validate a single checkpoint
- `ckptkit scan <root>`: validate all checkpoints under root
- `ckptkit resume <root>`: choose checkpoint to resume (policy configurable)
- `ckptkit quarantine <path>`: move checkpoint into `corrupt/` with reason
- `ckptkit emit-metrics`: write Prometheus textfile or push to Pushgateway

## Observability
- Metrics emitted: `checkpoint_last_success_timestamp`, `checkpoint_last_success_step`, `checkpoint_last_duration_seconds`, `checkpoint_write_bytes_total`, `checkpoint_validation_failures_total{reason}`, `checkpoint_corrupt_detected{checkpoint}`, `checkpoint_resume_selected_step`, `checkpoint_directory_free_bytes`
- Structured JSON logs include `run_id`, `job_id`, `step`, `checkpoint_path`, `event`, `severity`, `reason`
- Grafana dashboard JSON available at `dashboards/grafana_checkpoint_health.json`
- Prometheus scrape example at `examples/prometheus.yml`

## Testing
```bash
pytest
```

## Notes
- Atomic rename behavior may vary on network filesystems; prefer local disks or PVCs that preserve POSIX atomicity.
- Hashing large checkpoints can be expensive; use `--fast` validation to skip hashes or configure `--sample-bytes` for partial hashing.
