#!/usr/bin/env python3
"""tts.py — wrapper: Sanskrit .txt in, .mp3 out.

Reads a shloka from a text file, splits it into hemistichs on daṇḍa (।/॥) and
newline boundaries, drives src/render.py once for the whole batch, then stitches
the per-hemistich wavs into a single mp3 via ffmpeg.

    python scripts/tts.py verse.txt                      # -> ./verse.mp3
    python scripts/tts.py verse.txt -o /tmp/vande.mp3    # -> that exact path
    python scripts/tts.py verse.txt --format mp3         # only mp3 is wired up for now
    python scripts/tts.py verse.txt --meter vasantatilakā --nfe 64 --seed 42

Requires the venv from scripts/setup.sh and the BigVGAN clone. This script sets
PYTHONPATH for the render subprocess itself, so you don't need to export it.
"""
import argparse, json, os, re, subprocess, sys, tempfile, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RENDER = REPO / "src" / "render.py"
BIGVGAN = REPO / "BigVGAN"
DANDA_RE = re.compile(r"[।॥]+")

def split_hemistichs(text: str) -> list[str]:
    """Split on daṇḍas first, then on remaining newlines. Trim, drop empties."""
    pieces = []
    for chunk in DANDA_RE.split(text):
        for line in chunk.splitlines():
            line = line.strip()
            if line:
                pieces.append(line)
    return pieces

def build_shard(hemistichs: list[str], meter: str, seed: int, outdir: Path) -> list[dict]:
    return [
        {"id": f"clip_{i:03d}", "meter": meter, "padas": [h],
         "seed": seed, "no_sandhi": True,
         "out": str(outdir / f"clip_{i:03d}.wav")}
        for i, h in enumerate(hemistichs)
    ]

def run_render(shard_path: Path, outdir: Path, results_path: Path, nfe: int) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{BIGVGAN}{os.pathsep}{env.get('PYTHONPATH','')}"
    cmd = [sys.executable, str(RENDER),
           "--shard", str(shard_path),
           "--results", str(results_path),
           "--outdir", str(outdir),
           "--nfe", str(nfe)]
    subprocess.run(cmd, check=True, env=env, cwd=str(REPO))

def concat_to_mp3(wavs: list[Path], out_mp3: Path, gap_ms: int = 350) -> None:
    """ffmpeg concat with a short silent gap between hemistichs."""
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH — 'brew install ffmpeg'")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # generate a matching-sample-rate silence clip; probe rate from first wav
        rate = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate", "-of", "csv=p=0", str(wavs[0])]
        ).decode().strip()
        gap_wav = td / "gap.wav"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", f"anullsrc=r={rate}:cl=mono",
             "-t", f"{gap_ms/1000:.3f}", str(gap_wav)],
            check=True)
        listing = td / "concat.txt"
        with listing.open("w") as f:
            for i, w in enumerate(wavs):
                if i: f.write(f"file '{gap_wav.as_posix()}'\n")
                f.write(f"file '{w.resolve().as_posix()}'\n")
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "concat", "-safe", "0", "-i", str(listing),
             "-codec:a", "libmp3lame", "-qscale:a", "2",
             str(out_mp3)],
            check=True)

def main() -> None:
    ap = argparse.ArgumentParser(description="Sanskrit .txt -> chanted .mp3")
    ap.add_argument("input", type=Path, help="UTF-8 text file with a shloka")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output path (default: <input-stem>.mp3 in current dir)")
    ap.add_argument("--format", default="mp3", choices=["mp3"],
                    help="output format (only mp3 for now)")
    ap.add_argument("--meter", default="anushtubh",
                    help="meter key in src/reference_bank/bank.json (default: anushtubh)")
    ap.add_argument("--nfe", type=int, default=32,
                    help="flow-matching steps (32 serving, 64 locked)")
    ap.add_argument("--seed", type=int, default=60)
    ap.add_argument("--keep-wavs", action="store_true",
                    help="keep the per-hemistich wavs alongside the mp3")
    args = ap.parse_args()

    if not args.input.is_file():
        sys.exit(f"input not found: {args.input}")
    text = args.input.read_text(encoding="utf-8")
    hemistichs = split_hemistichs(text)
    if not hemistichs:
        sys.exit("no non-empty lines after daṇḍa/newline split")
    print(f"[tts] {len(hemistichs)} hemistich(s):")
    for i, h in enumerate(hemistichs):
        print(f"  [{i}] {h}")

    out_mp3 = args.output or Path.cwd() / f"{args.input.stem}.mp3"
    out_mp3 = out_mp3.resolve()
    out_mp3.parent.mkdir(parents=True, exist_ok=True)

    work = Path(tempfile.mkdtemp(prefix="tts_"))
    try:
        wav_dir = work / "wavs"; wav_dir.mkdir()
        shard = build_shard(hemistichs, args.meter, args.seed, wav_dir)
        shard_path = work / "shard.json"
        shard_path.write_text(json.dumps(shard, ensure_ascii=False, indent=2), encoding="utf-8")
        results_path = work / "results.json"
        run_render(shard_path, wav_dir, results_path, args.nfe)

        # verify every clip landed
        wavs = [Path(c["out"]) for c in shard]
        missing = [w for w in wavs if not w.exists()]
        if missing:
            sys.exit(f"render produced no wav for: {[str(m) for m in missing]} — see {results_path}")

        concat_to_mp3(wavs, out_mp3)

        if args.keep_wavs:
            keep_dir = out_mp3.parent / f"{out_mp3.stem}_wavs"
            keep_dir.mkdir(exist_ok=True)
            for w in wavs:
                shutil.copy2(w, keep_dir / w.name)
            print(f"[tts] kept wavs in {keep_dir}")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"[tts] wrote {out_mp3}")

if __name__ == "__main__":
    main()
