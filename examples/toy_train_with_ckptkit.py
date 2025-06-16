#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from ckptkit.atomic import atomic_checkpoint_write
from ckptkit.logging import log_event, setup_logging
from ckptkit.manifest import compute_manifest, manifest_path, write_manifest
from ckptkit.resume import Policy, select_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy training loop using ckptkit")
    parser.add_argument("--root", required=True, help="Checkpoint root directory")
    parser.add_argument("--job-id", default="toy-job")
    parser.add_argument("--run-id", default="run-1")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Attempt resume automatically")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logging("INFO")
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    start_step = 1
    if args.resume and any(root.iterdir()):
        try:
            plan = select_checkpoint(root, policy=Policy.LAST_KNOWN_GOOD)
            start_step = plan.step + 1
            log_event(
                logger,
                event="resume_plan",
                severity="INFO",
                checkpoint_path=str(plan.checkpoint),
                step=plan.step,
                job_id=args.job_id,
                run_id=args.run_id,
                reason=plan.reason,
            )
        except Exception as exc:
            log_event(logger, event="resume_failed", severity="WARNING", reason=str(exc))

    for step in range(start_step, args.steps + 1):
        loss = 1.0 / step + random.random() * 0.01
        time.sleep(0.05)

        def writer(tmp: Path):
            (tmp / "state.json").write_text(json.dumps({"step": step, "loss": loss}), encoding="utf-8")
            (tmp / "weights.bin").write_bytes(os.urandom(2048))
            manifest = compute_manifest(
                tmp,
                job_id=args.job_id,
                run_id=args.run_id,
                step=step,
                world_size=1,
                framework="pytorch",
            )
            write_manifest(manifest_path(tmp), manifest)
            return manifest

        dest = root / f"step-{step}"
        atomic_checkpoint_write(dest, writer)
        log_event(
            logger,
            event="checkpoint_written",
            severity="INFO",
            checkpoint_path=str(dest),
            step=step,
            job_id=args.job_id,
            run_id=args.run_id,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
