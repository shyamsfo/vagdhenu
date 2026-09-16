#!/usr/bin/env bash
# aws_bringup.sh — one-time OS prep on a freshly-launched GPU box.
#
# Runs FROM THE MAC. ssh's into the box and installs the OS-level bits that don't
# depend on the repo: apt packages (python3.10-venv, ffmpeg) and the `uv` installer.
# Idempotent — safe to re-run. Does NOT touch the repo or the venv; that's ship_to_box.sh.
#
# Usage (on your Mac):
#   deploy/aws_bringup.sh ubuntu@<box-ip>
#
# Prerequisites on the box:
#   - Deep Learning Base OSS NVIDIA Driver AMI (Ubuntu 22.04). python3.10 preinstalled.
#   - Your ssh key in ~ubuntu/.ssh/authorized_keys.
#
# Wall-clock: ~30–60 s. Follow with deploy/ship_to_box.sh <host>.
set -euo pipefail

HOST="${1:-}"
if [ -z "$HOST" ]; then
  echo "usage: $0 <user@host>" >&2
  exit 2
fi

ssh "$HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
echo "→ apt update + install python3.10-venv, ffmpeg"
sudo apt-get update -qq
sudo apt-get install -y -qq python3.10-venv ffmpeg

# uv installs to ~/.local/bin/uv. Ship script prepends this to PATH before use.
echo "→ install uv (if missing)"
if [ ! -x "$HOME/.local/bin/uv" ] && ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
"$HOME/.local/bin/uv" --version 2>/dev/null || uv --version

echo "✓ aws_bringup complete — next: deploy/ship_to_box.sh <host>"
REMOTE
