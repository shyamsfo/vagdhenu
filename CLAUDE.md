# CLAUDE.md — working notes for Vāgdhenu

Sanskrit chant (pārāyaṇa) TTS. A flow-matching DiT (IndicF5 / F5-TTS backbone, ~337M params)
renders metered ślokas, and a fine-tuned NVIDIA BigVGAN-v2 vocodes them. Full background is in
`docs/TECH_REPORT.md`; that document is the source of truth and this file is the short version.

## Branch topology

- **`main`** — the public Vāgdhenu tree (site + core model code).
- **`mac-port`** — the generic vagdhenu working branch. Mac-friendly `scripts/setup.sh` (uv +
  cu13 stragglers purge), Mac timings in `docs/MAC.md`, hardened `.gitignore`. No AWS or
  corpus-specific code.
- **`remote-render`** (this branch) — superset of `mac-port` that adds the Mac-driven AWS
  fan-out (`deploy/`) and the ganapati-sambhavam corpus glue (`scripts/ganapati_*.py`). Any
  change that is generic vagdhenu (model, text frontend, references, server, tts.py) belongs
  on `mac-port` and should be rebased into `main` from there. Anything that touches EC2 or
  a specific corpus stays here.

Do not commit generic vagdhenu changes on `remote-render` — they will strand here and drift
from `mac-port`. Switch branches, land the change on `mac-port`, then merge/rebase back.

## Hard invariants — do not "fix" these

These look like bugs. They are not. Each cost an experiment to establish.

1. **Sanskrit is routed through Kannada script, not Devanagari.** Devanagari input triggers Hindi
   schwa-deletion in the tokenizer. `src/prep_text.py` transliterates Deva → Kannada on the way in.
2. **`f5_tts` is AI4Bharat's IndicF5 fork, pinned to a commit in `requirements.txt`.** It is *not*
   the PyPI package `f5-tts`, which is different code whose `infer_process` crashes here.
3. **Locked inference parameters.** euler solver, nfe 64, cfg 3.0 (batch path), speed 0.90,
   sway −0.7, per-clip `fix_duration = ref_len + n_syll × sec_per_syll`. These came out of roughly
   80 numbered experiments. Changing one to make something "sound better" in isolation usually
   breaks a case that was deliberately tuned for. §19 of the tech report has the full table.
4. **The half-reference rule.** A reference entry's `ref_text` must match exactly what is spoken in
   its `.wav`, ending on a word or daṇḍa boundary. A clean half-hemistich around 7s beats a full
   15s śloka. F5 clips audio over 15s but keeps the full text, which garbles the output.
5. **Text-side swara embedding is a settled negative (E68).** F5 self-infilling already recovers
   pitch from the context mel, so the token carries no gradient. Do not re-attempt it.
6. **`src/render.py` is the frozen batch path.** It produced MBTN and the Bhāgavatam. Renders on
   CUDA must stay md5-identical across changes. Verify before and after any edit that touches it:
   `git stash`, render `examples/sample_shard.json`, `git stash pop`, render again, compare md5.

## Environment traps

- **`import bigvgan` needs `PYTHONPATH`.** BigVGAN is a cloned NVIDIA repo, not a pip package.
  `scripts/setup.sh` clones it; you must `export PYTHONPATH="$PWD/BigVGAN:$PYTHONPATH"` yourself.
- **`src/device.py` must be imported before `torch`.** Two reasons: (1) it arms
  `PYTORCH_ENABLE_MPS_FALLBACK`, which PyTorch reads exactly once when the Metal dispatch table
  registers at `import torch`; (2) on Linux, it prepends the venv's bundled `nvidia/cudnn/lib` and
  `nvidia/cublas/lib` to `LD_LIBRARY_PATH` so torch's dlopens find its matched-version bundled
  libs before the system copies. Without that prepend, the AWS Deep Learning AMI's cuDNN 9.10.2
  clashes with torch 2.4.1+cu121's bundled 9.24 and the first convolution fails with
  `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`. Every entry point imports `device` first on purpose.
  Do not tidy those import blocks and do not remove `_prepend_venv_cuda_libs()`.
- **Weights are gitignored** (`*.pt`, `*.pth`). Fetch with `python scripts/download_weights.py`,
  which lands them in `models/`. `CHAMP_ROOT` overrides that directory.
- **Audio is gitignored** except `examples/` and `src/reference_bank/`. Never commit a render.
- **The torch pin is re-asserted at the end of `scripts/setup.sh`.** `pip install torch==2.4.1` runs
  first, but `pip install -r requirements.txt` silently upgrades to torch 2.14 because IndicF5's
  own `setup.py` loosens to `torch>=2.0.0`. The tail of `setup.sh` reinstalls `torch==2.4.1
  torchaudio==2.4.1 --no-deps` and fails loudly if it slipped. Do not remove that block — 2.11+
  torchaudio requires torchcodec, which then needs Homebrew's ffmpeg dylibs on `@rpath`, which
  they aren't. This bit fresh installs on the GPU boxes too.
- **Three upper-bound pins in `requirements.txt` are load-bearing.** `starlette<1.0`,
  `pydantic<2.10`, and `huggingface_hub<1.0` each block a specific breakage in the demo server
  path (starlette 1.x kills Gradio's jinja templating with "unhashable type: 'dict'"; pydantic
  2.10 emits `additionalProperties: True` which crashes gradio_client 1.3.0's schema parser and
  turns `GET /` into a 500; hub 1.x conflicts with the pinned `transformers==4.46.3` and
  `tokenizers==0.20.3`). The comments in `requirements.txt` name each failure mode.
- Python 3.10 is the validated interpreter.

## Layout

```
src/prep_text.py      text frontend: Deva→SLP1→Kannada, visarga sandhi, anusvāra, daṇḍa rules
src/tts_meter.py      meter / gaṇa (laghu-guru) detection
src/render.py         frozen batch renderer (shard JSON in, wavs out) — the production path
src/render_core.py    the same pipeline as a reusable Renderer class, for interactive callers
src/device.py         backend selection + Linux LD_LIBRARY_PATH prepend — import before torch
src/limits.py         abuse guards (per-IP daily cap + one shloka per request); env-configurable
                      via VAGDHENU_DAILY_LIMIT / VAGDHENU_MAX_AKSHARAS (=0 disables each)
src/reference_bank/   ~27 reference wavs + bank.json, keyed by meter
demo/server.py        warm Gradio server, works on Mac/Linux/CUDA; env-configurable host + port
demo/app.py           HF ZeroGPU Space — GPU only, imports `spaces`
demo/README.md        three-mode deployment guide (local / dedicated GPU / HF Space)
scripts/selftest.py   staged smoke test; see docs/MAC.md
scripts/tts.py        driver: shloka .txt → .mp3 (splits daṇḍas, drives render.py, stitches ffmpeg)
scripts/ganapati_batch.py  ganapati-sambhavam corpus renderer (round-robin sharded)
scripts/ganapati_stitch.py post-run merge: per-sarga MP3s + merged manifest
scripts/README.md     per-script deep-dive; root README's Scripts section is the summary
deploy/               Mac-side AWS fan-out (launch / bring-up / ship / render / watch / terminate).
                      Every script runs on the Mac and ssh's into the box — see deploy/README.md.
docs/TECH_REPORT.md   the real documentation
docs/MAC.md           Mac/Metal setup + measured timings + regression-check recipe
```

## AWS fan-out

Six Mac-side scripts in `deploy/` drive the whole render on GPU boxes; you never ssh in by
hand. Invoke in numeric order: `launch_boxes.sh` → `aws_bringup.sh` → `ship_to_box.sh` →
`render_shard.sh` (detached) → `render_watch.sh` (polling) → `terminate_boxes.sh`. Full guide
in `deploy/README.md`. A few things that will bite:

- **`ship_to_box.sh` uses `--exclude-from=.gitignore` — not `--filter=':- .gitignore'`.**
  Dir-merge exclude rules do not participate in rsync's protect-from-`--delete` logic; the
  first version of this script wiped `BigVGAN/` off the box on every re-ship. Do not "clean
  up" back to the shorter filter form.
- **Explicit `--include` rules for `.wav`/`.mp3` under `src/reference_bank/` and `examples/`
  come *before* the `.gitignore` exclusion.** rsync applies rules in order, first match wins;
  git's `!pattern` negations are not honored by `--filter=':- .gitignore'` or
  `--exclude-from`, so the includes must be spelled out or the reference bank vanishes.
- **`render_shard.sh` detaches via `setsid <driver> </dev/null >log 2>&1 &`.** Do not switch
  to `nohup` alone (still receives SIGHUP under some sshd configs) and do not wait on the
  ssh session — the render is 2+ hours and you *will* disconnect. State is tracked via
  `.pid` / `.DONE` / `.FAILED` sentinels in `outputs/ganapati/`; `render_watch.sh` reads them.
- **`launch_boxes.sh` sets the SG ingress to your current public IP (via
  `checkip.amazonaws.com`).** If your IP moves between launch and ssh (VPN toggle, coffee
  shop, etc.) you need to add a fresh ingress rule — `deploy/README.md` has the command.
- **Instance type defaults to `g6e.xlarge` (L40S) but is `INSTANCE_TYPE`-overridable.** g6e
  hits regional capacity outages during US business hours; `g5.xlarge` (A10G, ~half cost) is
  the validated fallback and completes the ganapati corpus at RTF ~3.5.
- **No custom AMI** — we use the stock Deep Learning Base OSS NVIDIA GPU AMI (Ubuntu 22.04),
  which is why `src/device.py` has to prepend the venv's bundled cuDNN/cuBLAS on
  `LD_LIBRARY_PATH` (see the Environment traps section above).

## Testing

There is no unit test suite. `scripts/selftest.py` is the smoke harness: six stages from
environment checks through to a real-time-factor measurement. Stages 1 to 3 need no model weights.

```bash
python scripts/selftest.py --stages 1 2 3   # fast loop, no weights
python scripts/selftest.py --nfe 32         # full run
```

Numeric checks cannot tell you the chant is musically right. Listen to the output.

## Conventions

- Match the surrounding style. The render scripts are deliberately dense, with helpers copied
  verbatim between `render.py` and `render_core.py` so the two paths cannot drift.
- Comments explain *why* a value was chosen, usually with an experiment number. Keep that habit.
- Diacritics matter. Meter names and Sanskrit terms are written with them throughout.
