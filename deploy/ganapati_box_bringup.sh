#!/usr/bin/env bash
# ganapati_box_bringup.sh — bring one g6e.xlarge from fresh AMI to actively rendering.
#
# Run this ON THE GPU BOX, not locally. Copy it via scp or curl-raw from GitHub.
# Total wall-clock: ~15-20 min setup + ~1.7 hr render at nfe=64 for a 1/3 shard.
#
# Usage on the box:
#   bash ganapati_box_bringup.sh <SHARD_INDEX> <SHARD_COUNT>
# For the 3-box fan-out, run one of these on each box:
#   bash ganapati_box_bringup.sh 0 3     # box 1
#   bash ganapati_box_bringup.sh 1 3     # box 2
#   bash ganapati_box_bringup.sh 2 3     # box 3
#
# ── AWS launch (do this locally per box, before SSHing in) ─────────────────────
#
# Instance: g6e.xlarge (1× L40S, 24 GB VRAM), us-east-1, on-demand.
# AMI:      AWS Deep Learning Base OSS NVIDIA Driver GPU AMI (Ubuntu 22.04).
# SG:       inbound 22/tcp from your IP only. No public IP needed once rsync is done.
# Storage:  40 GB gp3 root is plenty (weights ~5 GB, outputs <500 MB per box).
#
#   AMI_ID=$(aws ssm get-parameter \
#     --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
#     --region us-east-1 --query 'Parameter.Value' --output text)
#
#   for i in 1 2 3; do
#     aws ec2 run-instances --region us-east-1 --instance-type g6e.xlarge \
#       --image-id "$AMI_ID" \
#       --key-name YOUR_KEY --security-group-ids sg-XXX \
#       --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=40,VolumeType=gp3}' \
#       --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ganapati-$i}]"
#   done
#
# ⚠ First-time on a new AWS account: `g6e` may need a quota bump for
# "Running On-Demand G and VT instances" (need at least 12 vCPU total for 3 boxes).
# Check with: aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA
#
# ── After launch ────────────────────────────────────────────────────────────────
# scp this script + your ssh into each box, then run it. Or embed as user-data.
#
set -euo pipefail

SHARD_INDEX="${1:-}"
SHARD_COUNT="${2:-3}"
if [[ -z "$SHARD_INDEX" ]]; then
  echo "usage: bash $0 <SHARD_INDEX> <SHARD_COUNT>" >&2
  exit 2
fi

# 1. System prep (~1 min)
sudo apt-get update -qq
sudo apt-get install -y -qq python3.10-venv ffmpeg

# 2. Clone repo (~10 s)
cd "$HOME"
if [[ ! -d vagdhenu ]]; then
  git clone https://github.com/shyamsfo/vagdhenu.git
fi
cd vagdhenu
git fetch origin && git checkout mac-port && git pull --ff-only

# 3. Venv + setup (~5-10 min: torch cu121 + f5_tts + BigVGAN clone + pinned deps)
if [[ ! -d .venv ]]; then
  python3.10 -m venv .venv
fi
source .venv/bin/activate
bash scripts/setup.sh

# 4. Download model weights (~5 min for ~5 GB)
python scripts/download_weights.py

# 5. Fetch input JSONs (~10 s — 500 KB total)
mkdir -p tmp/ganapati/shlokas
for i in 1 2 3 4 5 6 7 8 9 10; do
  curl -sfL "https://raw.githubusercontent.com/kolluruss/ganapati-sambhavam-site/main/shlokas/sarga-$i.json" \
    -o "tmp/ganapati/shlokas/sarga-$i.json"
done

# 6. Render this shard (~1.7 hr at nfe=64 on L40S for 1/3 of 849 shlokas)
export PYTHONPATH="$PWD/BigVGAN:${PYTHONPATH:-}"
mkdir -p outputs/ganapati
python scripts/ganapati_batch.py \
  --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" \
  --nfe 64 \
  --output-dir outputs/ganapati \
  2>&1 | tee "outputs/ganapati/render.shard-${SHARD_INDEX}-of-${SHARD_COUNT}.log"

echo
echo "=== Shard $SHARD_INDEX/$SHARD_COUNT done. ==="
echo "Output tree ready to rsync back:"
echo "  outputs/ganapati/shlokas/                       (per-shloka MP3s)"
echo "  outputs/ganapati/manifest.shard-${SHARD_INDEX}-of-${SHARD_COUNT}.json"
echo "  outputs/ganapati/preprocessing_report.shard-${SHARD_INDEX}-of-${SHARD_COUNT}.txt"
echo "  outputs/ganapati/render.shard-${SHARD_INDEX}-of-${SHARD_COUNT}.log"
echo
echo "From your Mac:"
echo "  rsync -av ubuntu@<this-box-ip>:vagdhenu/outputs/ganapati/ ./outputs/ganapati/"
echo "Then when all 3 boxes are drained:"
echo "  python scripts/ganapati_stitch.py"
