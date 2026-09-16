#!/usr/bin/env bash
# terminate_boxes.sh — kill instances tagged Name=<prefix>-* and delete the SG.
#
# Runs FROM THE MAC. Inverse of launch_boxes.sh. Idempotent: if the instances or the
# SG are already gone, exits cleanly. Waits for instances to fully terminate before
# deleting the SG (AWS refuses to delete a group that's still attached to any live
# or terminating ENI).
#
# Usage (on your Mac):
#   deploy/terminate_boxes.sh <name-prefix>
#
# Env:
#   AWS_REGION      default: us-east-1.
set -euo pipefail

PREFIX="${1:-}"
if [ -z "$PREFIX" ]; then
  echo "usage: $0 <name-prefix>" >&2
  exit 2
fi

REGION="${AWS_REGION:-us-east-1}"
SG_NAME="${PREFIX}-sg"

log() { echo "→ $*" >&2; }

log "find live instances tagged Name=$PREFIX-*"
INSTANCE_IDS=$(aws ec2 describe-instances --region "$REGION" \
  --filters "Name=tag:Name,Values=${PREFIX}-*" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].InstanceId' --output text)

if [ -z "$INSTANCE_IDS" ]; then
  log "no live instances for prefix '$PREFIX'"
else
  log "terminate: $INSTANCE_IDS"
  aws ec2 terminate-instances --region "$REGION" --instance-ids $INSTANCE_IDS >/dev/null
  log "wait for terminated state (~30–60 s)"
  aws ec2 wait instance-terminated --region "$REGION" --instance-ids $INSTANCE_IDS
fi

log "delete security group '$SG_NAME' (if present)"
SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
  --filters "Name=group-name,Values=$SG_NAME" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")
if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
  log "no security group named '$SG_NAME'"
else
  aws ec2 delete-security-group --region "$REGION" --group-id "$SG_ID"
  log "deleted $SG_ID"
fi

log "done"
