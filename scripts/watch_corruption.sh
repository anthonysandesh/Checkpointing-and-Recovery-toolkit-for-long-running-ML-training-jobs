#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-""}
INTERVAL=${2:-30}

if [[ -z "$ROOT" ]]; then
  echo "usage: $0 <checkpoint_root> [interval_seconds]" >&2
  exit 1
fi

STATE_FILE=$(mktemp)
trap "rm -f $STATE_FILE" EXIT

echo "[ckptkit] Watching $ROOT for corrupt checkpoints every $INTERVAL seconds"

while true; do
  OUTPUT=$(ckptkit scan "$ROOT" || true)
  echo "$OUTPUT"
  NEW_INVALID=$(echo "$OUTPUT" | grep "invalid" | awk '{print $1}')
  for ckpt in $NEW_INVALID; do
    if ! grep -Fxq "$ckpt" "$STATE_FILE"; then
      echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ") ALERT: new corrupt checkpoint detected $ckpt" >&2
      echo "$ckpt" >> "$STATE_FILE"
    fi
  done
  sleep "$INTERVAL"
done
