#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-""}
INTERVAL=${2:-60}

if [[ -z "$ROOT" ]]; then
  echo "usage: $0 <checkpoint_root> [interval_seconds]" >&2
  exit 1
fi

echo "[ckptkit] Starting validation loop for $ROOT every $INTERVAL seconds"

while true; do
  timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  if ckptkit scan "$ROOT" --sample-bytes 65536; then
    echo "$timestamp validation ok"
  else
    echo "$timestamp validation failed" >&2
  fi
  sleep "$INTERVAL"
done
