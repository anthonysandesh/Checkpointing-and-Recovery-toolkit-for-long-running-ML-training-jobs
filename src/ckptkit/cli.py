from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

from .atomic import atomic_checkpoint_write
from .config import Config, HashingConfig, RetentionConfig
from .logging import log_event, setup_logging
from .manifest import compute_manifest, manifest_path, write_manifest
from .metrics import MetricsEmitter, record_checkpoint_write, record_disk_free, record_resume_plan, record_validation_metrics
from .quarantine import quarantine as quarantine_ckpt
from .resume import Policy, select_checkpoint
from .validate import validate_checkpoint
from .fs import ensure_dir, list_checkpoints


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ckptkit", description="Checkpointing & Recovery Toolkit")
    parser.add_argument("--config", help="YAML config path", default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    sub = parser.add_subparsers(dest="command", required=True)

    write = sub.add_parser("write", help="Write a sample checkpoint")
    write.add_argument("--root", required=True, help="Checkpoint root directory")
    write.add_argument("--job-id", required=True)
    write.add_argument("--run-id", required=True)
    write.add_argument("--step", type=int, required=True)
    write.add_argument("--world-size", type=int, default=1)
    write.add_argument("--framework", default=None)
    write.add_argument("--precision", default=None)
    write.add_argument("--model-name", default=None)
    write.add_argument("--keep-last", type=int, default=None)
    write.add_argument("--keep-every", type=int, default=None)

    val = sub.add_parser("validate", help="Validate a single checkpoint")
    val.add_argument("path", help="Path to checkpoint directory")
    val.add_argument("--full", action="store_true", help="Full hash verification")
    val.add_argument("--sample-bytes", type=int, default=65536)

    scan = sub.add_parser("scan", help="Validate all checkpoints under root")
    scan.add_argument("root", help="Checkpoint root")
    scan.add_argument("--full", action="store_true")
    scan.add_argument("--sample-bytes", type=int, default=65536)

    resume_cmd = sub.add_parser("resume", help="Choose checkpoint to resume")
    resume_cmd.add_argument("root", help="Checkpoint root")
    resume_cmd.add_argument("--policy", choices=[p.value for p in Policy], default=Policy.LATEST_VALID.value)
    resume_cmd.add_argument("--before-step", type=int, default=None)
    resume_cmd.add_argument("--full", action="store_true")

    quarantine_cmd = sub.add_parser("quarantine", help="Quarantine a checkpoint")
    quarantine_cmd.add_argument("path", help="Path to checkpoint")
    quarantine_cmd.add_argument("--reason", required=True, help="Reason for quarantine")

    metrics_cmd = sub.add_parser("emit-metrics", help="Emit Prometheus metrics")
    metrics_cmd.add_argument("--root", required=True, help="Checkpoint root")
    metrics_cmd.add_argument("--textfile", help="Write metrics to textfile for node_exporter")
    metrics_cmd.add_argument("--pushgateway", help="Pushgateway base URL")
    metrics_cmd.add_argument("--job", default="ckptkit")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv) if argv is not None else sys.argv[1:])
    logger = setup_logging("DEBUG" if args.verbose else "INFO")

    if args.command == "write":
        retention = RetentionConfig(
            keep_last=args.keep_last if args.keep_last is not None else 3,
            keep_every=args.keep_every,
        )
        cfg = _load_config(
            args.config,
            {
                "root": args.root,
                "job_id": args.job_id,
                "run_id": args.run_id,
                "retention": dataclasses.asdict(retention),
            },
        )
        root = cfg.root
        ensure_dir(root)
        start = time.time()

        def writer(tmp: Path):
            # Demo checkpoint writer: two shards and metadata.
            shard1 = tmp / "model.bin"
            shard1.write_bytes(os.urandom(1024))
            shard2 = tmp / "optimizer.bin"
            shard2.write_bytes(os.urandom(512))
            meta = {
                "step": args.step,
                "job_id": args.job_id,
                "run_id": args.run_id,
            }
            (tmp / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            manifest = compute_manifest(
                tmp,
                job_id=args.job_id,
                run_id=args.run_id,
                step=args.step,
                world_size=args.world_size,
                framework=args.framework,
                precision=args.precision,
                model_name=args.model_name,
            )
            write_manifest(manifest_path(tmp), manifest)
            return manifest

        dest_dir = root / f"step-{args.step}"
        manifest = atomic_checkpoint_write(dest_dir, writer, retention=cfg.retention)
        duration = time.time() - start
        emitter = MetricsEmitter({"job_id": cfg.job_id, "run_id": cfg.run_id})
        total_bytes = sum(f.size for f in manifest.files)
        record_checkpoint_write(
            emitter,
            checkpoint_path=dest_dir,
            manifest_step=manifest.step,
            duration_seconds=duration,
            total_bytes=float(total_bytes),
        )
        log_event(
            logger,
            event="checkpoint_written",
            severity="INFO",
            checkpoint_path=str(dest_dir),
            step=args.step,
            job_id=cfg.job_id,
            run_id=cfg.run_id,
        )
        print(emitter.text())
        return 0

    if args.command == "validate":
        res = validate_checkpoint(Path(args.path), full_hash=args.full, sample_bytes=args.sample_bytes)
        print(res.summary())
        return 0 if res.valid else 1

    if args.command == "scan":
        cfg = _load_config(args.config, {"root": args.root})
        root = cfg.root
        results = [
            validate_checkpoint(
                ckpt, full_hash=args.full, sample_bytes=args.sample_bytes if args.sample_bytes else cfg.hashing.sample_bytes
            )
            for ckpt in list_checkpoints(root)
        ]
        for res in results:
            print(res.summary())
        invalid = [r for r in results if not r.valid]
        return 0 if not invalid else 1

    if args.command == "resume":
        cfg = _load_config(args.config, {"root": args.root})
        plan = select_checkpoint(
            cfg.root,
            policy=Policy(args.policy),
            before_step=args.before_step,
            full_hash=args.full,
        )
        print(json.dumps({"checkpoint": str(plan.checkpoint), "step": plan.step, "reason": plan.reason}))
        return 0

    if args.command == "quarantine":
        ckpt = Path(args.path)
        target = quarantine_ckpt(ckpt, root=ckpt.parent, reason=args.reason)
        print(f"quarantined to {target}")
        return 0

    if args.command == "emit-metrics":
        cfg = _load_config(args.config, {"root": args.root})
        root = cfg.root
        results = [validate_checkpoint(ckpt, full_hash=False, sample_bytes=65536) for ckpt in list_checkpoints(root)]
        plan = None
        if results:
            try:
                plan = select_checkpoint(root)
            except Exception:
                plan = None
        emitter = MetricsEmitter()
        record_validation_metrics(emitter, results)
        if plan:
            record_resume_plan(emitter, plan)
        record_disk_free(emitter, root)
        textfile_path = args.textfile or cfg.metrics.textfile
        pushgateway_url = args.pushgateway or cfg.metrics.pushgateway
        if textfile_path:
            emitter.write_textfile(Path(textfile_path))
        if pushgateway_url:
            emitter.push_gateway(pushgateway_url, job=args.job or cfg.metrics.pushgateway_job)
        print(emitter.text())
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


def _load_config(config_path: str | None, overrides: dict) -> Config:
    if config_path:
        return Config.load(config_path, overrides)
    return Config.from_dict(overrides)
