#!/usr/bin/env bash
# ship_to_box.sh — rsync this repo to the box, then create venv + install deps + weights.
#
# Runs FROM THE MAC. Re-runnable during iteration: rsync sends only diffs, and
# scripts/setup.sh is idempotent (torch pin, cu13 purge, BigVGAN clone, weight download
# all skip when already in place). Assumes deploy/aws_bringup.sh has been run once.
#
# Usage (on your Mac):
#   deploy/ship_to_box.sh ubuntu@<box-ip>
#
# rsync scope: everything not in .gitignore, minus .git/. That excludes .venv/, models/,
# BigVGAN/, outputs/, tmp/, and audio artifacts — none of which belong on the box from
# the Mac. --delete keeps the remote tree in lockstep with the Mac (excluded paths are
# preserved on the remote since we don't pass --delete-excluded).
#
# Wall-clock: ~10 min first run (setup.sh + ~5 GB weights), ~5 s subsequent iterations.
set -euo pipefail

HOST="${1:-}"
if [ -z "$HOST" ]; then
  echo "usage: $0 <user@host>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="vagdhenu"

echo "→ rsync $REPO_ROOT/ → $HOST:$REMOTE_DIR/"
# Two gotchas with git-style ignores under rsync, both surfaced by the initial
# end-to-end smoke test:
#   1. `--filter=':- .gitignore'` (dir-merge, exclude-only) does NOT protect
#      matching files on the receiver from --delete. This wipes BigVGAN/,
#      models/, .venv/ on re-runs. `--exclude-from` does protect (rsync docs
#      §"per-directory rules and --delete" hedge on dir-merge behavior).
#   2. Neither honors git's `!pattern` negation. `.gitignore` unignores three
#      paths via `!` (reference_bank wavs, examples/**/*.wav|mp3); those need
#      explicit --include rules, ordered BEFORE the exclude so they win.
# If new `!` rules land in .gitignore, add matching --include lines here.
rsync -a --delete \
  --include='src/reference_bank/*.wav' \
  --include='examples/**/*.wav' \
  --include='examples/**/*.mp3' \
  --exclude-from="$REPO_ROOT/.gitignore" \
  --exclude='.git/' \
  "$REPO_ROOT/" "$HOST:$REMOTE_DIR/"

echo "→ create venv + run scripts/setup.sh on box"
ssh "$HOST" 'bash -s' <<REMOTE
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
cd "\$HOME/$REMOTE_DIR"
if [ ! -d .venv ]; then
  uv venv --seed
fi
source .venv/bin/activate
bash scripts/setup.sh
REMOTE

echo "✓ ship_to_box complete — repo at $HOST:$REMOTE_DIR, venv ready"
