#!/usr/bin/env python3
"""ganapati_batch.py — render the kolluruss/ganapati-sambhavam-site corpus to per-shloka MP3s.

Loads all sarga-*.json files from --input-dir (each an array of {"number", "devanagari"}),
flattens to a stable ordered list of (sarga, shloka_number, text) entries, filters by
--shard-index / --shard-count (round-robin), preprocesses text (strips noise the frontend
would otherwise pass through: stray Latin letters, foreign Brahmic glyphs, ZWNJ; and logs
each drop), loads the Renderer ONCE, and emits <output-dir>/shlokas/sarga-NN-shloka-NNN.mp3
per entry. Writes per-shard manifest + preprocessing report next to the shlokas dir.

Deployment: run three copies with --shard-index 0/1/2 --shard-count 3 on three GPU boxes;
each box renders its ~283-shloka slice independently, then rsync all outputs to one host and
run scripts/ganapati_stitch.py to build the per-sarga concatenations + merged manifest.

    python scripts/ganapati_batch.py --shard-index 0 --shard-count 3     # box 1
    python scripts/ganapati_batch.py --shard-index 1 --shard-count 3     # box 2
    python scripts/ganapati_batch.py --shard-index 2 --shard-count 3     # box 3

Locked-quality defaults match render.py (nfe 64, cfg 3.0, speed 0.90, seed 60). Meter is
uniform shardulavikridita for this corpus (verified by inspection + partial detector runs).
"""
import argparse, glob, io, json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
BIGVGAN = REPO / "BigVGAN"
MODELS = Path(os.environ.get("CHAMP_ROOT", REPO / "models"))

# The frontend prep_text.strip_punct drops ASCII digits/punct/hyphens/apostrophes and Deva digits,
# but PASSES THROUGH stray Latin letters and non-Devanagari Brahmic glyphs unchanged — those then
# get mangled by sanscript's Deva→SLP1→Kannada transliteration and produce gibberish in the model
# input. Filter them here; log every drop so the source can be corrected upstream.
DEVA_LO, DEVA_HI = 0x0900, 0x097F                       # Devanagari block
ASCII_PUNCT_OK = set(" \t\r\n.,;:!?()\"'-|/\\")         # prep_text.strip_punct handles these
ASCII_DIGITS = set("0123456789")                        # prep_text drops these
DEVA_DANDA = set("।॥")                                  # already in Deva block, listed for clarity


def is_allowed(c: str) -> bool:
    o = ord(c)
    if DEVA_LO <= o <= DEVA_HI:                         # includes ०-९ and ।॥
        return True
    if c in ASCII_PUNCT_OK or c in ASCII_DIGITS:
        return True
    return False


def clean_text(raw: str) -> tuple[str, list[str]]:
    """Return (cleaned, dropped_chars). Preserves Devanagari + punct prep_text understands;
    drops everything else (Latin letters, foreign Brahmic glyphs, ZWNJ, misc)."""
    out, dropped = [], []
    for c in raw:
        if is_allowed(c):
            out.append(c)
        else:
            dropped.append(c)
    cleaned = "".join(out)
    # collapse any run of whitespace the drops opened up
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned, dropped


def load_corpus(input_dir: Path) -> list[dict]:
    """Return a stable-ordered list of {sarga, number, devanagari}."""
    files = sorted(input_dir.glob("sarga-*.json"),
                   key=lambda p: int(re.search(r"sarga-(\d+)", p.name).group(1)))
    if not files:
        sys.exit(f"no sarga-*.json files in {input_dir}")
    entries = []
    for f in files:
        sarga = int(re.search(r"sarga-(\d+)", f.name).group(1))
        for e in json.load(f.open(encoding="utf-8")):
            entries.append({"sarga": sarga, "number": int(e["number"]),
                            "devanagari": e["devanagari"]})
    entries.sort(key=lambda e: (e["sarga"], e["number"]))
    return entries


def write_mp3(audio, sr: int, out_mp3: Path, quality: str = "2") -> None:
    """Encode float32 [-1,1] audio to MP3 via a WAV -> ffmpeg pass. Uses libmp3lame -qscale:a 2
    to match scripts/tts.py's output quality."""
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not on PATH — 'brew install ffmpeg' / 'apt install ffmpeg'")
    import soundfile as sf
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "out.wav"
        sf.write(wav, audio, sr)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(wav), "-codec:a", "libmp3lame", "-qscale:a", quality,
             str(out_mp3)],
            check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-render ganapati-sambhavam shlokas to MP3.")
    ap.add_argument("--input-dir", type=Path, default=REPO / "tmp" / "ganapati" / "shlokas",
                    help="dir of sarga-*.json files (default: tmp/ganapati/shlokas)")
    ap.add_argument("--output-dir", type=Path, default=REPO / "outputs" / "ganapati",
                    help="where to write shlokas/ + manifest + report (default: outputs/ganapati)")
    ap.add_argument("--meter", default="shardulavikridita",
                    help="bank meter key for all shlokas (default: shardulavikridita)")
    ap.add_argument("--shard-index", type=int, default=0, help="0-based shard id")
    ap.add_argument("--shard-count", type=int, default=1, help="total shards (round-robin split)")
    ap.add_argument("--limit", type=int, default=0,
                    help="render at most N shlokas (0 = all in this shard); useful for smoke tests")
    ap.add_argument("--nfe", type=int, default=64, help="flow-matching steps (64 = locked)")
    ap.add_argument("--cfg", type=float, default=3.0, help="cfg strength (3.0 = locked batch)")
    ap.add_argument("--speed", type=float, default=0.90, help="F5 speed (0.90 = locked)")
    ap.add_argument("--seed", type=int, default=60, help="base seed (60 = MBTN marathon)")
    ap.add_argument("--voice", default=str(MODELS / "voice_steer_ema_2026-06-17.pt"))
    ap.add_argument("--voc", default=str(MODELS / "voc_bigvgan_EMA_2026-06-11.pth"))
    ap.add_argument("--bank", default=str(SRC / "reference_bank" / "bank.json"))
    ap.add_argument("--vocab", default=str(MODELS / "vocab.txt"))
    ap.add_argument("--no-resume", action="store_true",
                    help="re-render existing MP3s (default: skip)")
    args = ap.parse_args()

    if args.shard_count < 1 or not (0 <= args.shard_index < args.shard_count):
        sys.exit(f"invalid shard: index={args.shard_index} count={args.shard_count}")

    # Make BigVGAN importable (mirror scripts/tts.py's env handling)
    os.environ["PYTHONPATH"] = f"{BIGVGAN}{os.pathsep}{os.environ.get('PYTHONPATH','')}"
    sys.path.insert(0, str(BIGVGAN))
    sys.path.insert(0, str(SRC))

    # device is picked inside Renderer; import order matters (device.py before torch — handled
    # by render_core.py itself). Loading torch here is fine because render_core already
    # imported device at module top.
    from render_core import Renderer

    entries = load_corpus(args.input_dir)
    total = len(entries)
    # Round-robin: box K takes indices where idx % shard_count == shard_index.
    my_entries = [e for i, e in enumerate(entries)
                  if i % args.shard_count == args.shard_index]
    if args.limit > 0:
        my_entries = my_entries[:args.limit]
    print(f"[ganapati] shard {args.shard_index+1}/{args.shard_count}: "
          f"{len(my_entries)} of {total} shlokas")

    out_dir = args.output_dir
    shloka_dir = out_dir / "shlokas"
    shloka_dir.mkdir(parents=True, exist_ok=True)
    tag = f"shard-{args.shard_index}-of-{args.shard_count}"
    manifest_path = out_dir / f"manifest.{tag}.json"
    report_path = out_dir / f"preprocessing_report.{tag}.txt"

    print(f"[ganapati] loading Renderer (voice={Path(args.voice).name}, "
          f"nfe={args.nfe}, cfg={args.cfg}, speed={args.speed})")
    t0 = time.time()
    r = Renderer(args.voice, args.voc, args.bank,
                 vocab_file=args.vocab, speed=args.speed, nfe=args.nfe, cfg=args.cfg)
    print(f"[ganapati] Renderer ready in {time.time()-t0:.1f}s on device={r.device}")

    # Verify meter is in the bank (fail fast rather than silently falling back per shloka).
    # Renderer accepts either raw name (e.g. 'śārdūlavikrīḍita') or wav-stem alias
    # ('shardulavikridita') — both are keyed in _lut.
    if args.meter.lower() not in r._lut:
        sys.exit(f"meter '{args.meter}' not in bank. Available raw keys: {r.meters()}")

    manifest, dropped_report = [], []
    n_rendered, n_skipped, n_failed = 0, 0, 0
    t_start = time.time()

    for i, e in enumerate(my_entries):
        stem = f"sarga-{e['sarga']:02d}-shloka-{e['number']:03d}"
        mp3_path = shloka_dir / f"{stem}.mp3"
        cleaned, dropped = clean_text(e["devanagari"])
        if dropped:
            dropped_report.append((stem, dropped, e["devanagari"]))

        if not args.no_resume and mp3_path.exists():
            n_skipped += 1
            manifest.append({"stem": stem, "sarga": e["sarga"], "number": e["number"],
                             "mp3": str(mp3_path.relative_to(out_dir)),
                             "devanagari": e["devanagari"], "cleaned": cleaned,
                             "dropped_chars": dropped, "status": "resumed"})
            continue

        try:
            t_clip = time.time()
            sr, audio = r.render_one(cleaned, meter=args.meter, seed=args.seed)
            write_mp3(audio, sr, mp3_path)
            dt = time.time() - t_clip
            dur = len(audio) / sr
            n_rendered += 1
            manifest.append({"stem": stem, "sarga": e["sarga"], "number": e["number"],
                             "mp3": str(mp3_path.relative_to(out_dir)),
                             "devanagari": e["devanagari"], "cleaned": cleaned,
                             "dropped_chars": dropped,
                             "duration_s": round(dur, 3), "render_s": round(dt, 2),
                             "status": "rendered"})
            eta = (time.time() - t_start) / max(1, n_rendered) * (len(my_entries) - i - 1)
            print(f"[{i+1}/{len(my_entries)}] {stem}: {dur:.1f}s audio / "
                  f"{dt:.1f}s compute (RTF {dt/max(dur,0.01):.2f})  eta {eta/60:.1f}m")
        except Exception as ex:
            n_failed += 1
            manifest.append({"stem": stem, "sarga": e["sarga"], "number": e["number"],
                             "devanagari": e["devanagari"], "cleaned": cleaned,
                             "dropped_chars": dropped, "status": "failed",
                             "error": f"{type(ex).__name__}: {ex}"})
            print(f"[{i+1}/{len(my_entries)}] {stem}: FAILED — {ex}", file=sys.stderr)

    manifest_path.write_text(
        json.dumps({"shard_index": args.shard_index, "shard_count": args.shard_count,
                    "meter": args.meter, "nfe": args.nfe, "cfg": args.cfg,
                    "speed": args.speed, "seed": args.seed,
                    "voice": Path(args.voice).name, "voc": Path(args.voc).name,
                    "rendered": n_rendered, "skipped": n_skipped, "failed": n_failed,
                    "entries": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"# Preprocessing report — shard {args.shard_index+1}/{args.shard_count}\n")
        f.write(f"# {len(dropped_report)} of {len(my_entries)} shlokas had chars stripped.\n\n")
        for stem, drops, orig in dropped_report:
            uniq = sorted(set(drops))
            f.write(f"{stem}: {len(drops)} char(s) dropped: {uniq!r}\n")
            f.write(f"  original: {orig[:120].replace(chr(10),' / ')}...\n\n")

    dt_total = time.time() - t_start
    print(f"[ganapati] done: {n_rendered} rendered, {n_skipped} skipped, {n_failed} failed "
          f"in {dt_total/60:.1f}m ({dt_total/max(1,n_rendered):.1f}s/clip avg)")
    print(f"[ganapati] manifest: {manifest_path}")
    print(f"[ganapati] report:   {report_path}")


if __name__ == "__main__":
    main()
