#!/usr/bin/env bash
# launch_boxes.sh — spin up N g6e.xlarge GPU boxes and print their public IPs.
#
# Runs FROM THE MAC. Creates a fresh security group named "<prefix>-sg" with a single
# rule (inbound 22/tcp from your current public IP), tags each instance with Name=
# <prefix>-<i>, and blocks until every box has passed both EC2 status checks (that's
# when ssh actually works). Prints one IP per line on stdout; also writes to
# tmp/last-launch.txt for later recovery.
#
# Usage (on your Mac):
#   deploy/launch_boxes.sh <count> <name-prefix>
#   # Example — one-liner for the ganapati fan-out:
#   IPS=$(deploy/launch_boxes.sh 3 ganapati)
#   for ip in $IPS; do deploy/aws_bringup.sh ubuntu@$ip & done; wait
#   for ip in $IPS; do deploy/ship_to_box.sh ubuntu@$ip & done; wait
#
# Env:
#   AWS_KEY_NAME    ssh keypair registered in EC2. Default: id_nuwire.
#   AWS_REGION      default: us-east-1.
#   INSTANCE_TYPE   default: g6e.xlarge (1× L40S, 24 GB VRAM). Use g5.xlarge (A10G,
#                   ~half cost, ~2× slower) as a fallback when g6e is capacity-out.
#
# AMI:      AWS Deep Learning Base OSS NVIDIA Driver GPU AMI (Ubuntu 22.04), SSM-resolved.
# Root:     AMI default (75 GB gp3 — the AMI's baked snapshot is that size, can't shrink).
# Wall-clock: ~2 min from run-instances to ssh-reachable.
#
# ⚠ Quota: g6e instances count against "Running On-Demand G and VT instances" (each
# g6e.xlarge = 4 vCPU). Fresh AWS accounts default to 0. Bump with:
#   aws service-quotas request-service-quota-increase \
#     --service-code ec2 --quota-code L-DB2E81BA --desired-value 16
set -euo pipefail

COUNT="${1:-}"
PREFIX="${2:-}"
if [ -z "$COUNT" ] || [ -z "$PREFIX" ]; then
  echo "usage: $0 <count> <name-prefix>" >&2
  exit 2
fi

REGION="${AWS_REGION:-us-east-1}"
KEY_NAME="${AWS_KEY_NAME:-id_nuwire}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g6e.xlarge}"
SG_NAME="${PREFIX}-sg"

log() { echo "→ $*" >&2; }

log "resolve AMI (SSM)"
AMI_ID=$(aws ssm get-parameter --region "$REGION" \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
  --query 'Parameter.Value' --output text)
echo "   AMI_ID=$AMI_ID" >&2

log "verify keypair '$KEY_NAME' exists in $REGION"
aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY_NAME" >/dev/null

log "detect my public IP (checkip.amazonaws.com)"
MY_IP=$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')
echo "   MY_IP=$MY_IP" >&2

log "create security group '$SG_NAME'"
SG_ID=$(aws ec2 create-security-group --region "$REGION" \
  --group-name "$SG_NAME" --description "vagdhenu $PREFIX ssh access" \
  --query 'GroupId' --output text)
echo "   SG_ID=$SG_ID" >&2

log "authorize inbound 22/tcp from $MY_IP/32"
aws ec2 authorize-security-group-ingress --region "$REGION" \
  --group-id "$SG_ID" --protocol tcp --port 22 --cidr "$MY_IP/32" >/dev/null

log "launch $COUNT × $INSTANCE_TYPE (Name=$PREFIX-<i>, SG=$SG_NAME)"
INSTANCE_IDS=()
for i in $(seq 1 "$COUNT"); do
  IID=$(aws ec2 run-instances --region "$REGION" \
    --instance-type "$INSTANCE_TYPE" \
    --image-id "$AMI_ID" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$PREFIX-$i}]" \
    --count 1 \
    --query 'Instances[0].InstanceId' --output text)
  INSTANCE_IDS+=("$IID")
  echo "   [$i] $IID" >&2
done

log "wait for instance-status-ok on all ${#INSTANCE_IDS[@]} boxes (~2 min)"
aws ec2 wait instance-status-ok --region "$REGION" --instance-ids "${INSTANCE_IDS[@]}"

log "fetch public IPs"
mkdir -p tmp
: > tmp/last-launch.txt
for IID in "${INSTANCE_IDS[@]}"; do
  IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  echo "$IP"                                     # stdout: one IP per line
  echo "$IID $IP" >> tmp/last-launch.txt         # side channel for recovery
done

log "ready. instance-id + IP recorded in tmp/last-launch.txt"
log "when done: deploy/terminate_boxes.sh $PREFIX"
