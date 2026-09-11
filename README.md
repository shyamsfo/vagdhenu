# Vāgdhenu — Sanskrit Chant TTS

*"The wish-cow of speech."* A production-grade, single-speaker **Sanskrit chant (pārāyaṇa) text-to-speech** system — it *chants* classical ślokas with metrically-aware durations and tradition-faithful melodic contour, not flat read-aloud.

> **MOS ~4.6** (expert listener). Conjuncts — including retroflex aspirates (ṣṭ, ḍḍh, …) — render 100% correctly, the class earlier architectures could not crack. Used to produce **MBTN** (32 YouTube videos, 17h 34m) and the **Śrīmad Bhāgavatam** (16,017 verses, audio app + 31 karaoke videos).

[ **[Project page + live demo](https://prathosh.in/vagdhenu/)** · [Model weights → HF](https://huggingface.co/prathoshap/vagdhenu) · [Demo → HF Space](https://huggingface.co/spaces/prathoshap/vagdhenu-demo) · Tech report → `docs/TECH_REPORT.md` ]

## Demos (rendered with this system)
- **Mahābhārata Tātparya Nirṇaya (MBTN)** — full chant series: [YouTube playlist](https://www.youtube.com/playlist?list=PLL1s8qiaGy0IP0G_PhlwaGA5EOfzoKrV_)
- **Śrīmad Bhāgavatam** — karaoke-video series: [YouTube playlist](https://www.youtube.com/playlist?list=PLDiYyVdyo2Sc)

Developed and maintained by **Prof. Prathosh, Indian Institute of Science, Bengaluru.**

## How it works
- **Backbone:** IndicF5 / F5-TTS — a flow-matching **DiT** (OT-CFM mel-infilling, ~337M params, *no* native duration or pitch head). Sanskrit is routed through **Kannada script** (Devanagari triggers Hindi schwa-deletion).
- **Vocoder:** NVIDIA **BigVGAN-v2**, fine-tuned on F5 vocos-mel (mandatory — vocos shivers on long vowels).
- **Prosody:** F5's content fidelity is bulletproof but its prosody is *text-driven, not designable*. The working levers are **the reference clip** (voice + swara + pace, via the *half-reference rule*) and a **voice-steering fine-tune**. (See `docs/TECH_REPORT.md` §14 for the full account — this is the central architectural finding.)
- **Text frontend (`src/prep_text.py`)** — the most reusable piece: Deva→SLP1→Kannada routing, internal visarga sandhi (utva/rutva/lopa/satva), homorganic anusvāra, vocalic-ṝ handling, daṇḍa-final rules, meter/gaṇa (L/G) detection.

## Layout
```
src/         text frontend, meter detection, inference, post-gate, reference bank
pipeline/    data-prep (cut→pair→train) + build/assemble/QC
demo/        Gradio app (HF ZeroGPU)
docs/        scrubbed technical report + frontend/pipeline references
examples/    sample inputs + rendered outputs
scripts/     env setup, weight download, selftest harness, tts driver
```

## Install & quickstart
Requires **Python 3.10**. Production target is a **CUDA 12.1 GPU**; Apple Silicon (Metal/MPS) and
CPU also run — the backend is auto-detected, see *Running on a Mac* below.
```bash
bash scripts/setup.sh    # torch (cu121 on Linux / arm64 on macOS), deps, BigVGAN, weights -> models/
# render a Devanagari verse (+ meter) to a chanted wav:
python src/render.py --shard examples/sample_shard.json --results /tmp/res.json --outdir out
# -> out/sample_anushtubh.wav
```
The batch renderer takes a shard JSON: `[{"id","meter","padas":[devanagari…],"seed","out"}]`. For one-off single-verse renders see `src/render_production.py`. `CHAMP_ROOT` env overrides the weights dir (default `models/`).

### Running on a Mac
Full setup and run order: **[`docs/MAC.md`](docs/MAC.md)**.

Apple Silicon only — PyTorch has shipped no macOS x86_64 wheel since 2.2.2, so Intel Macs cannot run this.
`src/device.py` picks the backend (`cuda` → `mps` → `cpu`) and arms `PYTORCH_ENABLE_MPS_FALLBACK`
before torch loads, so the mel/ISTFT FFT kernels fall back to CPU on torch builds whose Metal
coverage is incomplete instead of raising. Pin a backend by hand with `VAGDHENU_DEVICE=cpu|mps|cuda`.

Expect it to be slow. On an A6000 the locked `nfe 64` config renders at RTF 1.24, `nfe 32` at 0.63.
A modern 16-core x86 CPU renders anuṣṭubh at nfe 64 at RTF ~23 (about 1.5 min per hemistich) — the
tech report's "~22 min per hemistich" figure was measured on much older hardware and is far too
pessimistic. Apple Metal is verified end-to-end for anuṣṭubh at nfe 32 (all five hot kernels run
native, no CPU fallback); wall-time RTF is unmeasured but expected to land between A6000 and CPU.
Peak inference memory is 2.5 GB, so any Apple Silicon Mac has room. Also `brew install ffmpeg`
(pydub decodes the reference clips). **Renders will not be bit-identical to the GPU output** —
different backend, different kernel order — so the md5-identical guarantee holds on CUDA only.
`demo/app.py` stays GPU-only (it imports HF ZeroGPU's `spaces`); use `demo/server.py` locally.

## Scripts

Four entrypoints live in `scripts/`. Run them from the repo root with the venv active.

- **`scripts/setup.sh`** — one-shot bootstrap: installs torch + deps, clones NVIDIA BigVGAN, downloads weights into `models/`. Branches on `uname` so the same command works on Linux (CUDA 12.1) and Apple Silicon (Metal). Refuses Intel Macs.
- **`scripts/download_weights.py`** — fetches the Vāgdhenu weights and the IndicF5 vocab into `$CHAMP_ROOT` (default `models/`). Called by `setup.sh`; run it directly to refresh.
- **`scripts/selftest.py`** — six-stage smoke harness (environment → kernel coverage → text frontend → single render → batch render → mel-distance comparison). Stages 1–3 need no weights. See `docs/MAC.md` for the full flow.
- **`scripts/tts.py`** — driver: takes a UTF-8 text file of one or more ślokas, splits on daṇḍas + newlines, drives `src/render.py`, and stitches the hemistich wavs into an MP3.
  ```
  python scripts/tts.py verse.txt                    # -> ./verse.mp3
  python scripts/tts.py verse.txt -o /tmp/x.mp3 --nfe 32 --seed 42
  ```

## Case studies
- **MBTN** (Mahābhārata Tātparya Nirṇaya) — 32-adhyāya *video* deliverable (Devanagari + Kannada karaoke, tanpura), shipped.
- **Śrīmad Bhāgavatam** — 12 skandhas, ~18k verses, *audio* app + a 31-video 3-script (Devanāgarī · Kannada · IAST) karaoke series. Sanskrit text gratefully acknowledged to **Poornaprajna Samshodhana Mandiram, Bengaluru**.

## Attribution & licenses
- Code: **Apache-2.0** (`LICENSE`).
- Built on **AI4Bharat IndicF5** (MIT), **NVIDIA BigVGAN-v2**, and **F5-TTS** — see their licenses; weights redistributed per those terms.
- Model weights + intended-use/ethics note: see the HF model card.

## Ethics / intended use
Single-speaker synthesis of sacred Sanskrit recitation, for pārāyaṇa/study/accessibility. The voice is the author's own. Please use responsibly; do not impersonate.

## Citation
*(BibTeX added with the arXiv report.)*
