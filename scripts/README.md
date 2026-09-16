# `scripts/`

Six entrypoints. Run them from the **repo root** with the venv active (`source .venv/bin/activate`)
and `PYTHONPATH="$PWD/BigVGAN:$PYTHONPATH"` exported. `tts.py` and `ganapati_batch.py` set
`PYTHONPATH` for their own subprocess, but the others assume you've done it yourself.

Rough mental model: **`setup.sh`** gets you a working install (it calls **`download_weights.py`**);
**`selftest.py`** proves the install works end-to-end; **`tts.py`** is the user-facing driver for
ad-hoc verses. **`ganapati_batch.py`** + **`ganapati_stitch.py`** are the corpus-specific pair
driven by `deploy/render_shard.sh` when fanning out to three GPU boxes — see `deploy/README.md`.

---

## `setup.sh`

One-shot environment bootstrap. ~40 lines of bash. Runs `pip` inside whichever venv you have
active; does not create one for you.

**What it does**, in order:
1. Branches on `uname`:
   - **Linux** → `pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121` (CUDA 12.1 wheels).
   - **macOS arm64** → same versions from stock PyPI (the arm64 wheels ship the Metal backend). Warns if `ffmpeg` isn't on `PATH`.
   - **macOS x86_64** → refuses with an explanation. PyTorch stopped publishing macOS x86_64 wheels after 2.2.2, so Intel Macs cannot run this.
2. `pip install -r requirements.txt` — installs IndicF5 (from its GitHub commit pin), transformers, vocos, x-transformers, librosa, etc.
3. Clones NVIDIA BigVGAN into `BigVGAN/` (it's a repo, not a pip package — no `setup.py` there).
4. Runs `scripts/download_weights.py` (see below).
5. **Re-asserts the torch pin.** `pip install -r requirements.txt` above will silently upgrade torch to 2.14 and torchaudio to 2.11 (via IndicF5's `torch>=2.0.0` constraint). This step reinstalls `torch==2.4.1 torchaudio==2.4.1 --no-deps` and fails loudly if the version slipped. Do not remove — see `CLAUDE.md` for why.
6. Prints the resolved backend by calling into `src/device.py`, so you can eyeball whether you got `cuda`, `mps`, or `cpu`.

**When to run:** once, after `git clone`. Re-run only if you want to refresh dependencies or you deleted `.venv/`.

**Doesn't do:** create the venv, export `PYTHONPATH`, install ffmpeg. Those are on you.

---

## `download_weights.py`

Sixteen lines. Uses `huggingface_hub.hf_hub_download` to pull four files from `prathoshap/vagdhenu` on Hugging Face into `models/`:

| File | What it is |
|---|---|
| `voice_steer_ema_2026-06-17.pt` | DiT voice-steering fine-tune — the current production voice |
| `voice_armA_ema_2026-06-11.pt` | Alternate voice checkpoint (the "armA" variant); not used by default renders |
| `voc_bigvgan_EMA_2026-06-11.pth` | Fine-tuned BigVGAN-v2 vocoder |
| `vocab.txt` | IndicF5's tokenizer vocab, redistributed so you don't depend on the gated `ai4bharat/IndicF5` repo |

**Env vars:**
- `VAGDHENU_HF` — override the HF repo path (default `prathoshap/vagdhenu`).
- `CHAMP_ROOT` — override where files land (default `models/`, used by `src/render.py`).

**When to run:** called by `setup.sh`. Run standalone to re-download after `rm -rf models/` or when the checkpoints get updated on HF.

---

## `selftest.py`

Six-stage smoke harness (~340 lines). Each stage prints PASS/FAIL; `--stages 1 2 3` runs a subset.

| Stage | Needs weights? | What it checks |
|---|---|---|
| 1. Environment & backend | no | Python version, torch version, platform, which backend `device.py` picked, `PYTORCH_ENABLE_MPS_FALLBACK` armed, `ffmpeg` on `PATH`, `bigvgan` importable, voice + vocoder + bank + vocab files all present |
| 2. Kernel coverage | no | Re-runs 5 hot kernels in a **child process with the MPS fallback disabled**, so anything Metal actually lacks raises instead of silently falling back to CPU. Per-kernel verdict: `native` or `CPU-FALLBACK`. Kernels probed: `torch.stft` (mel front-end), `torch.istft` (vocos head), `torchaudio.MelSpectrogram`, `scaled_dot_product_attention` (DiT), `ConvTranspose1d` (BigVGAN). |
| 3. Text frontend digest | no | Runs `prep_text.py` on a canonical Devanagari input and prints a hash. Device-independent — the digest must match across platforms, catches accidental changes to the Deva → SLP1 → Kannada routing / visarga sandhi / anusvāra / daṇḍa rules. |
| 4. Single-shard render | **yes** | One clip through the full pipeline end-to-end. |
| 5. Batch render | **yes** | Multiple clips, exercises the model-loaded-once loop path used by `src/render.py`. |
| 6. Mel-distance compare | **yes** | With `--compare path/to/reference.wav`, measures spectral distance to a reference produced on another backend. Mel-distance, not md5 — the only comparison that means anything across backends (see `docs/MAC.md`, "Known differences from the GPU output"). |

**Flags:**
- `--stages 1 2 3` — run a subset.
- `--nfe 32` (default 64) — flow-matching steps for the render stages.
- `--outdir` — where renders land (default `out/`).
- `--compare path.wav` — reference wav for stage 6.

**When to run:** stages 1–3 are cheap and need no weights, so run them anytime you want a smoke test. Stages 4–6 for verifying that a new machine or a code change didn't break the render.

---

## `tts.py`

User-facing driver — the reason non-developers care about `scripts/`. Reads a UTF-8 shloka text
file, splits it into verses (on `॥`) and hemistichs (on `।` + newlines), drives `src/render.py`,
and stitches the per-hemistich wavs into MP3(s) via `ffmpeg`.

```bash
python scripts/tts.py verse.txt                       # -> ./verse.mp3
python scripts/tts.py verse.txt -o /tmp/vande.mp3     # exact output path
python scripts/tts.py verse.txt --nfe 64 --seed 42    # override render params
python scripts/tts.py verse.txt --keep-wavs           # keep the raw hemistichs
```

**Flags:**
- `-o` / `--output` — output path (default: `<input-stem>.mp3` in current dir). With `--chunk-size`, this is a base pattern: `<stem>_001.mp3`, `<stem>_002.mp3`, …
- `--format` — output format, `mp3` only for now.
- `--meter` — reference-bank key (default `anushtubh`). Must match a key in `src/reference_bank/bank.json`. Also the fallback for verses `--auto-meter` can't classify.
- `--auto-meter` — detect meter per verse via `src/tts_meter.py`. Needs a full 4-pāda verse (or 32 syllables for anuṣṭubh); unrecognized verses fall through to `--meter`.
- `--chunk-size N` — emit one MP3 per N verses instead of one giant file. Recommended for anything over a few dozen verses: makes failures locally recoverable and produces playable-sized files.
- `--resume` — with `--chunk-size`, skip chunks whose output MP3 already exists. Rerun the same command after a failure to pick up where it left off.
- `--nfe` — flow-matching steps (default 32 = serving; 64 = locked production config).
- `--seed` — per-clip seed (default 60). If a clip's RMS energy falls below threshold, `render.py` will advance the seed up to 4 times before giving up on that clip.
- `--keep-wavs` — copy the per-hemistich wavs into `<output-stem>_wavs/` alongside each mp3.

**Notes:**
- Sets `PYTHONPATH` for the render subprocess itself, so you don't need to export it just to run this.
- Verse boundaries prefer `॥`; when the input has none, blank-line separated blocks are treated as verses. Verse-number lines (`१`, `1`) are dropped automatically.
- Resume is per-chunk, not per-hemistich: a chunk that fails mid-render restarts from its first hemistich on rerun. Pick a chunk size that keeps individual-chunk wall time tolerable (e.g. 10–20 anuṣṭubhs).

**Scaling to large texts** (hundreds+ verses): use `--chunk-size` + `--resume` directly. One command handles the loop, and a failure mid-way is safe to rerun:

```bash
python scripts/tts.py bhagavatam.txt --chunk-size 20 --nfe 64 -o bhagavatam.mp3 --resume
# -> bhagavatam_001.mp3 ... bhagavatam_040.mp3
# if verse #937 fails, fix and rerun the same line — completed chunks are skipped
```

For mixed-meter corpora add `--auto-meter`; a summary line prints how many verses fell back to `--meter`.

`render.py` continues past failures (retries seeds up to 4×, then gives up on that clip), so **check the results after** — a shard with `FAIL=3/40` won't be obvious from listening to the mp3.

---

## `ganapati_batch.py`

Corpus-specific renderer for `kolluruss/ganapati-sambhavam-site` (10 sargas, 847 ślokas, all
śārdūlavikrīḍita). Loads all `sarga-*.json` from `--input-dir`, flattens to a stable order,
round-robin shards on `--shard-index / --shard-count`, and renders each entry to
`<output-dir>/shlokas/sarga-NN-shloka-NNN.mp3`. Loads the Renderer once; skips existing MP3s
by default (`--no-resume` to force re-render). Writes per-shard `manifest.shard-<i>-of-<n>.json`
+ `preprocessing_report.shard-<i>-of-<n>.txt` next to the shlokas dir.

Meter is hardcoded to `shardulavikridita` — verified by inspection of the source. Locked-quality
defaults match `render.py` (nfe 64, cfg 3.0, speed 0.90, seed 60).

**Pre-processing filter:** `prep_text.strip_punct` passes stray Latin letters and non-Devanagari
Brahmic glyphs through to the sanscript Deva→SLP1→Kannada step, which then produces gibberish.
This script pre-filters anything outside the Devanagari block + ASCII punct/digits and logs
every drop into the shard's preprocessing report. Check the report; if a specific shloka has
many drops, the source needs fixing upstream.

**When to run:** by hand for single-shard smoke tests
(`python scripts/ganapati_batch.py --shard-index 0 --shard-count 1 --limit 3`); otherwise via
`deploy/render_shard.sh`, which handles fetching the sarga JSONs, detaching via `setsid`, and
writing sentinel files that `deploy/render_watch.sh` polls.

---

## `ganapati_stitch.py`

Post-processing after all shards' outputs have been rsync'd back to one host (`render_watch.sh`
does the rsync incrementally). Merges `manifest.shard-*.json` into `manifest.json`,
ffmpeg-concats the per-shloka MP3s into per-sarga MP3s (`sarga-01.mp3` through `sarga-10.mp3`)
with a 900 ms silent gap between shlokas, and produces a single `preprocessing_report.txt`.

Idempotent — re-run whenever new shards land or a shloka is re-rendered. Missing shlokas are
logged as `missing_file`, not fatal. Verifies render params match across shards and warns on
mismatch.

```bash
python scripts/ganapati_stitch.py                            # defaults to outputs/ganapati
python scripts/ganapati_stitch.py --output-dir /path/to/out  # custom location
```

**When to run:** after all three shards' `render_watch.sh` reports `DONE` and one final
`rsync` pass has landed the shloka MP3s locally.

---

For the higher-level project overview see the top-level [`README.md`](../README.md). For the Mac-specific
setup and run order, see [`../docs/MAC.md`](../docs/MAC.md). For the Mac→AWS render pipeline
see [`../deploy/README.md`](../deploy/README.md).
