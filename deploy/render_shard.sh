#!/usr/bin/env bash
# render_shard.sh — kick off a detached ganapati shard render on a prepared box.
#
# Runs FROM THE MAC. ssh's into the box, fetches the 10 sarga JSONs synchronously,
# writes a per-shard driver script, then launches it via setsid so the render
# survives the ssh disconnect. Returns in ~15 s. Poll with deploy/render_watch.sh.
#
# Assumes deploy/ship_to_box.sh has been run against this host — the repo is at
# ~/vagdhenu, the venv is at ~/vagdhenu/.venv, and weights are in ~/vagdhenu/models.
#
# Remote state files (in ~/vagdhenu/outputs/ganapati/):
#   render.shard-N-of-M.pid         PID of the detached bash driver
#   render.shard-N-of-M.log         stdout + stderr
#   render.shard-N-of-M.DONE        touched on exit 0
#   render.shard-N-of-M.FAILED      contains the exit code on non-zero
#   render.shard-N-of-M.driver.sh   the per-shard launcher (idempotent)
#
# Usage (on your Mac):
#   deploy/render_shard.sh <user@host> <shard-index> <shard-count>
#   # Fan-out (returns quickly, watch progress separately):
#   i=1; for ip in $IPS; do
#     deploy/render_shard.sh ubuntu@$ip $i 3
#     i=$((i+1))
#   done
#   watch -n 600 "deploy/render_watch.sh $(for ip in $IPS; do printf 'ubuntu@%s ' $ip; done)"
#
# Env (all optional):
#   NFE               inference steps per shloka. Default 64.
#   OUTPUT_DIR        relative to ~/vagdhenu on the box. Default outputs/ganapati.
#
# Wall-clock at nfe=64 on L40S: ~1.7 hr for 1/3 of the ~849-shloka corpus.
set -euo pipefail

HOST="${1:-}"
SHARD_INDEX="${2:-}"
SHARD_COUNT="${3:-}"
if [ -z "$HOST" ] || [ -z "$SHARD_INDEX" ] || [ -z "$SHARD_COUNT" ]; then
  echo "usage: $0 <user@host> <shard-index> <shard-count>" >&2
  exit 2
fi

NFE="${NFE:-64}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ganapati}"

ssh "$HOST" \
  "SHARD_INDEX='$SHARD_INDEX' SHARD_COUNT='$SHARD_COUNT' NFE='$NFE' OUTPUT_DIR='$OUTPUT_DIR' bash -s" \
  <<'REMOTE'
set -euo pipefail
cd "$HOME/vagdhenu"

TAG="shard-${SHARD_INDEX}-of-${SHARD_COUNT}"
PID_FILE="$OUTPUT_DIR/render.$TAG.pid"
LOG_FILE="$OUTPUT_DIR/render.$TAG.log"
DONE_FILE="$OUTPUT_DIR/render.$TAG.DONE"
FAIL_FILE="$OUTPUT_DIR/render.$TAG.FAILED"
DRIVER_FILE="$OUTPUT_DIR/render.$TAG.driver.sh"

mkdir -p "$OUTPUT_DIR" tmp/ganapati/shlokas

# Bail if a render is already running for this shard. Rare false positive if the
# PID got recycled after a reboot — delete the pid file manually to override.
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "✗ render already running for $TAG (pid $(cat "$PID_FILE")). Delete $PID_FILE to override." >&2
  exit 1
fi

# Clear previous DONE/FAILED sentinels — they refer to a prior run.
rm -f "$DONE_FILE" "$FAIL_FILE"

# Fetch input JSONs synchronously — cheap (~500 KB total, ~10 s). Must complete
# before python starts; better to fail here than crash the detached render.
echo "→ fetch 10 sarga JSONs from kolluruss/ganapati-sambhavam-site"
for i in 1 2 3 4 5 6 7 8 9 10; do
  curl -sfL "https://raw.githubusercontent.com/kolluruss/ganapati-sambhavam-site/main/shlokas/sarga-$i.json" \
    -o "tmp/ganapati/shlokas/sarga-$i.json"
done

# Write per-shard driver script. The outer heredoc (<<'REMOTE') is quoted so no
# Mac-side expansion; this inner heredoc (<<DRIVER) is unquoted so remote bash
# bakes in the shard/NFE/paths and preserves \$-escaped refs for the driver
# itself to expand at run time.
cat > "$DRIVER_FILE" <<DRIVER
#!/usr/bin/env bash
cd "\$HOME/vagdhenu"
source .venv/bin/activate
export PYTHONPATH="\$PWD/BigVGAN:\${PYTHONPATH:-}"
python scripts/ganapati_batch.py \\
  --shard-index $SHARD_INDEX --shard-count $SHARD_COUNT \\
  --nfe $NFE \\
  --output-dir "$OUTPUT_DIR"
ec=\$?
if [ \$ec -eq 0 ]; then touch "$DONE_FILE"; else echo \$ec > "$FAIL_FILE"; fi
exit \$ec
DRIVER
chmod +x "$DRIVER_FILE"

# Launch detached. setsid → new session so ssh's SIGHUP doesn't cascade;
# </dev/null → no stdin; >log 2>&1 → capture everything; & → background.
echo "→ launch detached render (shard $SHARD_INDEX/$SHARD_COUNT at nfe=$NFE)"
setsid "$DRIVER_FILE" </dev/null >"$LOG_FILE" 2>&1 &
RENDER_PID=$!
echo $RENDER_PID > "$PID_FILE"
disown $RENDER_PID 2>/dev/null || true

echo "✓ started $TAG (pid $RENDER_PID, log $LOG_FILE)"
REMOTE

echo "✓ $HOST: shard $SHARD_INDEX/$SHARD_COUNT running. Poll with deploy/render_watch.sh $HOST"
