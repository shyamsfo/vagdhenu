---
title: Vāgdhenu — Sanskrit Chant TTS
emoji: 🎶
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: demo/app.py
pinned: false
license: apache-2.0
short_description: Metered Sanskrit/Vedic chant text-to-speech (Devanagari in, audio out)
---

# Vāgdhenu — Sanskrit chant TTS (interactive UI)

The interactive web UI: paste a shloka in any Indic script, get chanted audio back. Same underlying
render pipeline as `src/render.py`, wrapped in Gradio.

Two entry points live in this directory, and they support **three deployment modes** between them:

| Mode | Entry point | Where the model runs | GPU cost | Suits |
|---|---|---|---|---|
| Local | `demo/server.py` | Wherever you run it (Mac / dev box / rented GPU VM) | Your own hardware | Personal use, offline pārāyaṇa, development |
| Dedicated GPU server | `demo/server.py` | An always-on GPU box you own or rent | Full-time GPU | Public-facing UI without per-request cold start |
| Hugging Face Space | `demo/app.py` | HF ZeroGPU pool (H100, allocated on demand) | Free tier + per-visitor quota | Public demo with zero recurring cost |

Both scripts serve the same Gradio UI (one shloka in, chanted audio out; meter auto-detect;
seed hidden under Advanced). The choice is about hosting, not features.

---

## Mode 1 — Local

Run everything on your own machine. The whole pipeline stays offline.

```bash
git clone git@github.com:shyamsfo/vagdhenu.git && cd vagdhenu
python3.10 -m venv .venv && source .venv/bin/activate     # or `uv venv --seed .venv`
bash scripts/setup.sh                                      # torch, deps, BigVGAN, weights
export PYTHONPATH="$PWD/BigVGAN:$PYTHONPATH"

# On macOS: brew install ffmpeg   (only needed if you also want scripts/tts.py to stitch mp3s)

# Personal use: disable the per-IP daily cap and the single-shloka-per-request check
export VAGDHENU_DAILY_LIMIT=0
export VAGDHENU_MAX_AKSHARAS=0

python demo/server.py
# -> Open http://127.0.0.1:7860 in a browser
```

Backend auto-detected: CUDA if you have it, MPS on Apple Silicon, CPU otherwise. Model loads once
at startup (~30–90 s depending on backend) and stays warm for the life of the process.

**Notes:**
- Default bind is `127.0.0.1:7860`. To expose on your LAN, set `VAGDHENU_HOST=0.0.0.0`.
- Without `VAGDHENU_DAILY_LIMIT=0`, you'll hit the 10-render/day cap after 10 requests — because
  all your local browser calls come from `127.0.0.1`, which counts as one "IP." The check exists
  to protect the public demo from abuse; it makes no sense when the server is just for you.
- `MAX_AKSHARAS=0` lets you paste longer input. Without it, anything over ~100 aksharas or with
  more than one full daṇḍa is rejected as "more than one shloka."

---

## Mode 2 — Dedicated GPU server (on-prem or cloud VM)

Same script as Mode 1, but on a machine with a dedicated GPU that stays warm 24/7. This is what
runs behind `prathosh.in/vagdhenu/` on Prof. Prathosh's on-prem A6000, and it's what you'd deploy
on a rented cloud GPU (AWS g4dn, Lambda Labs, RunPod, etc.).

```bash
# same setup steps as Mode 1
export VAGDHENU_HOST=0.0.0.0             # bind on all interfaces for LAN / public access
# keep the default limits (VAGDHENU_DAILY_LIMIT=10, VAGDHENU_MAX_AKSHARAS=100)
# — or bump them, e.g. VAGDHENU_DAILY_LIMIT=50 for a trusted user base
python demo/server.py &
```

**Notes:**
- `demo/server.py` loads the model *once at process boot* and holds it warm. Every request reuses
  the loaded model — no per-request cold start. This is why a dedicated GPU is worth the recurring
  cost for a public service.
- Rate limits (`limits.py`) are per-IP and per-day. In-memory, resets on process restart. Fine for
  a demo; not a durable quota system. If you need something stronger, put nginx / Cloudflare
  in front and rate-limit there.
- Add a reverse proxy (Caddy, nginx, Cloudflare Tunnel) for TLS and DNS. `server.py` speaks plain
  HTTP; don't expose it directly.

---

## Mode 3 — Hugging Face Space (ZeroGPU)

Zero recurring cost. A pool of H100 GPUs is allocated on demand when a visitor triggers the
`@spaces.GPU`-decorated function; between requests, no GPU is held. The tradeoff is per-request
cold start latency (~10–20 s while ZeroGPU allocates).

```bash
huggingface-cli login    # once
huggingface-cli upload prathoshap/vagdhenu-demo . --repo-type space
```

The `demo/README.md` frontmatter above (`sdk: gradio`, `sdk_version`, `app_file: demo/app.py`) is
what HF Spaces reads to configure the runtime. HF Spaces provides gradio via its SDK, so
`demo/requirements.txt` (not the root `requirements.txt`) is what governs deps for the Space.

**Notes:**
- The Space imports `spaces` (the ZeroGPU decorator package), which only exists in HF's runtime —
  don't try to run `demo/app.py` locally, use `demo/server.py` instead.
- First request downloads the weights from `prathoshap/vagdhenu`; subsequent requests reuse them
  (they're cached in the Space's persistent storage).
- Visitors get a daily GPU-second quota from HF. HF Pro subscribers get more; free-tier visitors
  get less. If a visitor blows through, the widget shows a rate-limit error.
- The Space **also** honors `VAGDHENU_DAILY_LIMIT` and `VAGDHENU_MAX_AKSHARAS` via `limits.py` —
  set them in the Space's environment settings if you want to loosen the defaults.

---

## Environment variables

| Variable | Default | Applies to | Meaning |
|---|---|---|---|
| `VAGDHENU_HOST` | `127.0.0.1` (server.py) | `demo/server.py` | Bind address. Set to `0.0.0.0` to expose on LAN / public. |
| `VAGDHENU_PORT` | `7860` | `demo/server.py` | Port. |
| `VAGDHENU_NFE` | `32` | `demo/server.py` | Flow-matching steps (64 = locked production, 32 = serving). |
| `VAGDHENU_VOICE` | resolved from `models/` or `demo/weights/` | both | Path to the voice-steering `.pt`. |
| `VAGDHENU_VOC` | resolved from `models/` or the on-prem path | both | Path to the BigVGAN `.pth`. |
| `VAGDHENU_HF` | `prathoshap/vagdhenu` | `demo/app.py` | HF repo for weight downloads on the Space. |
| `VAGDHENU_DEVICE` | auto (`cuda`→`mps`→`cpu`) | both | Force a backend (`cpu`, `mps`, or `cuda`). |
| `VAGDHENU_DAILY_LIMIT` | `10` | `src/limits.py` (both) | Per-IP daily render cap. **`0` disables the cap.** |
| `VAGDHENU_MAX_AKSHARAS` | `100` | `src/limits.py` (both) | Reject requests with more aksharas than this. **`0` disables the check.** |

## Notes

- First request is slow on the Space (weight download + GPU allocation). Local warm servers pay
  the load cost once at boot and every request is fast after that.
- The synthesis pipeline mirrors the gold batch renderer (`src/render.py`); see `src/render_core.py`.
- For non-interactive batch conversion (text file → mp3), use `scripts/tts.py` instead of the UI.
