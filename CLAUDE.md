# CLAUDE.md — working notes for Vāgdhenu

Sanskrit chant (pārāyaṇa) TTS. A flow-matching DiT (IndicF5 / F5-TTS backbone, ~337M params)
renders metered ślokas, and a fine-tuned NVIDIA BigVGAN-v2 vocodes them. Full background is in
`docs/TECH_REPORT.md`; that document is the source of truth and this file is the short version.

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
- **`src/device.py` must be imported before `torch`.** It arms `PYTORCH_ENABLE_MPS_FALLBACK`, and
  PyTorch reads that variable once, when the Metal dispatch table registers at `import torch`.
  Every entry point imports it first on purpose. Do not tidy those import blocks.
- **Weights are gitignored** (`*.pt`, `*.pth`). Fetch with `python scripts/download_weights.py`,
  which lands them in `models/`. `CHAMP_ROOT` overrides that directory.
- **Audio is gitignored** except `examples/` and `src/reference_bank/`. Never commit a render.
- **The torch pin is re-asserted at the end of `scripts/setup.sh`.** `pip install torch==2.4.1` runs
  first, but `pip install -r requirements.txt` silently upgrades to torch 2.14 because IndicF5's
  own `setup.py` loosens to `torch>=2.0.0`. The tail of `setup.sh` reinstalls `torch==2.4.1
  torchaudio==2.4.1 --no-deps` and fails loudly if it slipped. Do not remove that block — 2.11+
  torchaudio requires torchcodec, which then needs Homebrew's ffmpeg dylibs on `@rpath`, which
  they aren't. This bit fresh installs on the GPU boxes too.
- Python 3.10 is the validated interpreter.

## Layout

```
src/prep_text.py      text frontend: Deva→SLP1→Kannada, visarga sandhi, anusvāra, daṇḍa rules
src/tts_meter.py      meter / gaṇa (laghu-guru) detection
src/render.py         frozen batch renderer (shard JSON in, wavs out) — the production path
src/render_core.py    the same pipeline as a reusable Renderer class, for interactive callers
src/device.py         backend selection (cuda / mps / cpu) — import before torch
src/reference_bank/   ~27 reference wavs + bank.json, keyed by meter
demo/server.py        standalone warm Gradio server (loads the model once)
demo/app.py           HF ZeroGPU Space — GPU only, imports `spaces`
scripts/selftest.py   staged smoke test; see docs/MAC.md
scripts/tts.py        driver: shloka .txt → .mp3 (splits daṇḍas, drives render.py, stitches ffmpeg)
docs/TECH_REPORT.md   the real documentation
```

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
