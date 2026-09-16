# deploy/

Mac-side scripts for running vagdhenu renders on AWS GPU boxes. Every script
runs on the Mac and ssh's into the box for the actual work — you never need to
log into the box by hand.

## Scripts

Invoke in the order below.

| # | Script | What it does |
|---|--------|--------------|
| 0 | `launch_boxes.sh <count> <prefix>` | Create a fresh security group (inbound 22/tcp from your current public IP), launch N × `g6e.xlarge` (AMI-default 75 GB gp3 root) tagged `<prefix>-<i>`, wait for `instance-status-ok`, print public IPs on stdout (also mirrored to `tmp/last-launch.txt`). |
| 1 | `aws_bringup.sh <host>` | Per box: apt-install `python3.10-venv` + `ffmpeg`, install `uv` to `~/.local/bin`. Idempotent — safe to re-run. |
| 2 | `ship_to_box.sh <host>` | Per box: rsync repo (respecting `.gitignore`, excluding `.git/`), create `.venv` via `uv venv --seed`, run `scripts/setup.sh` (torch pin, cu13 purge, BigVGAN clone, weight download). Re-runnable — designed for the code-iteration loop. |
| 3 | `render_shard.sh <host> <i> <n>` | Per box: fetch 10 sarga JSONs, launch `scripts/ganapati_batch.py` **detached** via `setsid` so it survives ssh disconnect. Returns in ~15 s. Writes `render.shard-<i>-of-<n>.{pid,log,DONE,FAILED,driver.sh}` under `outputs/ganapati/` on the box. |
| 4 | `render_watch.sh <host>...` | One-shot: for each host, snapshot state (`RUNNING` / `DONE` / `FAILED` / `DEAD`) + rendered mp3 count + last log line, then `rsync -a` outputs back to `<repo>/outputs/ganapati/`. Wrap in `watch -n 600 …` or a `while … sleep 600` loop for periodic monitoring. |
| 5 | `terminate_boxes.sh <prefix>` | Terminate every instance tagged `<prefix>-*`, wait for `instance-terminated`, delete the `<prefix>-sg` security group. Idempotent. |

## End-to-end (ganapati fan-out)

```bash
# 0. launch
IPS=$(deploy/launch_boxes.sh 3 ganapati)
HOSTS=$(for ip in $IPS; do printf 'ubuntu@%s ' $ip; done)

# 1-2. OS prep + code/deps in parallel across boxes
for ip in $IPS; do deploy/aws_bringup.sh ubuntu@$ip & done; wait
for ip in $IPS; do deploy/ship_to_box.sh ubuntu@$ip & done; wait

# 3. detached renders (each call returns in ~15 s)
i=1; for ip in $IPS; do deploy/render_shard.sh ubuntu@$ip $i 3; i=$((i+1)); done

# 4. monitor + incremental sync every 10 min (in another terminal)
watch -n 600 "deploy/render_watch.sh $HOSTS"

# 5. when all show DONE
python scripts/ganapati_stitch.py     # combines shards on the Mac
deploy/terminate_boxes.sh ganapati
```

Steps 0–2 and 5 are generic vagdhenu infrastructure — they'll work for any
corpus. Steps 3–4 are ganapati-specific today (`render_shard.sh` hardcodes the
`kolluruss/ganapati-sambhavam-site` JSON URLs and calls `ganapati_batch.py`).
Genericization is deferred until a second corpus lands — see `to-do.md`.

## Prerequisites

- **AWS credentials** — `~/.aws/credentials` or `AWS_*` env vars; picked up by the AWS CLI automatically.
- **EC2 keypair** — defaults to `id_nuwire`, override via `AWS_KEY_NAME`.
- **Region** — defaults to `us-east-1`, override via `AWS_REGION`. `g6e` quota needs a bump on fresh accounts; the header of `launch_boxes.sh` has the `service-quotas` command.
- **ssh** — the Deep Learning Base OSS NVIDIA GPU AMI (Ubuntu 22.04) accepts your EC2 keypair under `~ubuntu/`, so `ssh ubuntu@<ip>` just works once the SG allows you in.

## Env vars

| Var | Default | Used by |
|-----|---------|---------|
| `AWS_KEY_NAME` | `id_nuwire` | `launch_boxes.sh` |
| `AWS_REGION` | `us-east-1` | `launch_boxes.sh`, `terminate_boxes.sh` |
| `INSTANCE_TYPE` | `g6e.xlarge` | `launch_boxes.sh` — fall back to `g5.xlarge` when g6e is capacity-out |
| `NFE` | `64` | `render_shard.sh` (inference steps per shloka) |
| `OUTPUT_DIR` | `outputs/ganapati` | `render_shard.sh` (relative to `~/vagdhenu` on the box) |

## Notes

- **SG is locked to your public IP at launch** (via `checkip.amazonaws.com`). If your IP changes between launch and ssh, add a rule:
  ```bash
  aws ec2 authorize-security-group-ingress --group-id <sg> --protocol tcp --port 22 --cidr <new-ip>/32
  ```
- **`render_shard.sh` refuses to double-start** — if a pid file exists and `kill -0` succeeds, it exits. Delete the pid file on the box to override.
- **State machine per shard** (from `render_watch.sh`): `DONE` → touch of `.DONE`; `FAILED (exit N)` → non-zero exit written to `.FAILED`; `RUNNING` → pid alive; `DEAD` → pid file exists but process gone with no sentinel (SIGKILL, OOM, etc. — check the log).
