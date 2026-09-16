#!/usr/bin/env bash
# render_watch.sh — one-shot status snapshot + incremental rsync from each box.
#
# Runs FROM THE MAC. For each host: one brief ssh to gather status, then rsync
# outputs back. Idempotent, read-mostly (touches only local outputs/). Wrap in a
# loop or `watch` for periodic monitoring.
#
# Usage (on your Mac):
#   deploy/render_watch.sh <user@host> [<user@host>...]
#
#   # Every 10 min, ctrl-C clean:
#   while true; do deploy/render_watch.sh $HOSTS; sleep 600; done
#
#   # Or `watch` for a live redraw:
#   watch -n 600 "deploy/render_watch.sh $HOSTS"
#
# State machine per shard (from render_shard.sh's sentinel files):
#   DONE                          → touch of render.<tag>.DONE. Ready to stitch.
#   FAILED (exit N)               → non-zero exit; check the log.
#   RUNNING (pid P)               → pid file exists and kill -0 succeeds.
#   DEAD                          → pid file exists but process is gone and no
#                                   DONE/FAILED marker — an ungraceful death,
#                                   check the log.
#   NO_ACTIVE_SHARD               → no pid file on the box.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <user@host> [<user@host>...]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_DIR="$REPO_ROOT/outputs/ganapati"
mkdir -p "$LOCAL_DIR"

for HOST in "$@"; do
  echo "── $HOST ──"

  # One ssh call gathers a 4-line status block: TAG / STATE / RENDERED / LOG_TAIL.
  OUTPUT=$(ssh "$HOST" 'bash -s' <<'REMOTE'
cd "$HOME/vagdhenu/outputs/ganapati" 2>/dev/null || { echo "NO_OUTPUT_DIR"; echo "-"; echo "0"; echo "-"; exit 0; }
PID_FILE=$(ls render.shard-*-of-*.pid 2>/dev/null | head -n 1)
if [ -z "$PID_FILE" ]; then
  echo "NO_ACTIVE_SHARD"; echo "-"; echo "0"; echo "-"; exit 0
fi
TAG=$(basename "$PID_FILE" .pid | sed 's/^render\.//')
PID=$(cat "$PID_FILE")
if [ -f "render.$TAG.DONE" ]; then
  STATE="DONE"
elif [ -f "render.$TAG.FAILED" ]; then
  STATE="FAILED (exit $(cat "render.$TAG.FAILED"))"
elif kill -0 "$PID" 2>/dev/null; then
  STATE="RUNNING (pid $PID)"
else
  STATE="DEAD (pid $PID gone, no DONE/FAILED — check log)"
fi
RENDERED=$(ls shlokas/*.mp3 2>/dev/null | wc -l | tr -d ' ')
LOG_TAIL=$(grep -v '^$' "render.$TAG.log" 2>/dev/null | tail -n 1)
echo "$TAG"
echo "$STATE"
echo "$RENDERED"
echo "${LOG_TAIL:--}"
REMOTE
)

  { read -r TAG; read -r STATE; read -r RENDERED; read -r LOG_TAIL; } <<< "$OUTPUT"

  echo "  shard:    $TAG"
  echo "  state:    $STATE"
  echo "  rendered: $RENDERED mp3s"
  echo "  tail:     $LOG_TAIL"

  # Incremental rsync — only new/changed files transfer. Bring back logs +
  # sentinels + mp3s + manifest fragments; leave source JSONs (they're upstream).
  echo "  → rsync outputs → $LOCAL_DIR/"
  rsync -a "$HOST:vagdhenu/outputs/ganapati/" "$LOCAL_DIR/"
  echo
done
