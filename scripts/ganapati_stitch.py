#!/usr/bin/env python3
"""ganapati_stitch.py — post-processing after all 3 GPU boxes' outputs have been rsync'd.

Merges the per-shard manifests into a single manifest.json, ffmpeg-concats the per-shloka
MP3s into per-sarga MP3s (sarga-01.mp3 through sarga-10.mp3), and produces a single
preprocessing_report.txt covering every entry that had characters stripped.

    python scripts/ganapati_stitch.py                        # defaults to outputs/ganapati
    python scripts/ganapati_stitch.py --output-dir /path/to/outputs/ganapati

Idempotent — safe to re-run once more shards arrive; existing per-sarga MP3s are re-built
from whatever is present on disk. Missing shlokas are logged, not fatal.
"""
import argparse, glob, json, re, shutil, subprocess, sys, tempfile
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent
GAP_MS = 900   # inter-shloka gap; longer than tts.py's 350ms since these are full shlokas, not hemistichs


def concat_mp3(mp3s: list[Path], out_mp3: Path, gap_ms: int = GAP_MS) -> None:
    """ffmpeg concat with a short silent gap between shlokas."""
    if not mp3s:
        return
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        sys.exit("ffmpeg/ffprobe not on PATH — 'brew install ffmpeg' / 'apt install ffmpeg'")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rate = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate", "-of", "csv=p=0", str(mp3s[0])]
        ).decode().strip()
        gap_wav = td / "gap.wav"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", f"anullsrc=r={rate}:cl=mono",
             "-t", f"{gap_ms/1000:.3f}", str(gap_wav)],
            check=True)
        listing = td / "concat.txt"
        with listing.open("w") as f:
            for i, m in enumerate(mp3s):
                if i:
                    f.write(f"file '{gap_wav.as_posix()}'\n")
                f.write(f"file '{m.resolve().as_posix()}'\n")
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "concat", "-safe", "0", "-i", str(listing),
             "-codec:a", "libmp3lame", "-qscale:a", "2",
             str(out_mp3)],
            check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge shard manifests + build per-sarga MP3s.")
    ap.add_argument("--output-dir", type=Path, default=REPO / "outputs" / "ganapati")
    ap.add_argument("--per-sarga", action="store_true", default=True,
                    help="build per-sarga MP3s (default on)")
    ap.add_argument("--no-per-sarga", dest="per_sarga", action="store_false")
    args = ap.parse_args()

    out_dir = args.output_dir
    shloka_dir = out_dir / "shlokas"
    if not shloka_dir.is_dir():
        sys.exit(f"no shlokas/ dir under {out_dir} — nothing to stitch")

    # Merge shard manifests
    shard_files = sorted(out_dir.glob("manifest.shard-*.json"))
    if not shard_files:
        sys.exit(f"no manifest.shard-*.json under {out_dir}")
    all_entries = []
    meta = {}
    for sf in shard_files:
        d = json.load(sf.open(encoding="utf-8"))
        # capture render params from the first shard; verify they match across shards
        if not meta:
            meta = {k: d[k] for k in ("meter", "nfe", "cfg", "speed", "seed", "voice", "voc")}
        else:
            mismatch = [k for k in meta if meta[k] != d.get(k)]
            if mismatch:
                print(f"[stitch] WARN {sf.name}: param mismatch {mismatch}", file=sys.stderr)
        all_entries.extend(d["entries"])

    # De-dupe on stem (in case of overlapping shards) and sort
    by_stem = {e["stem"]: e for e in all_entries}
    all_entries = sorted(by_stem.values(), key=lambda e: (e["sarga"], e["number"]))

    # Verify each entry's MP3 actually exists
    for e in all_entries:
        if e.get("status") in ("rendered", "resumed"):
            p = out_dir / e["mp3"]
            if not p.exists():
                e["status"] = "missing_file"
                print(f"[stitch] WARN missing file for {e['stem']}: {p}", file=sys.stderr)

    merged = {"meta": meta,
              "count": len(all_entries),
              "rendered": sum(1 for e in all_entries if e.get("status") in ("rendered", "resumed")),
              "missing":  sum(1 for e in all_entries if e.get("status") not in ("rendered", "resumed")),
              "entries": all_entries}
    (out_dir / "manifest.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[stitch] merged manifest: {merged['rendered']} rendered, {merged['missing']} missing")

    # Merged preprocessing report
    report_lines = ["# Preprocessing report — merged across all shards", ""]
    dropped_entries = [e for e in all_entries if e.get("dropped_chars")]
    report_lines.append(f"# {len(dropped_entries)} of {len(all_entries)} shlokas had chars stripped.\n")
    for e in dropped_entries:
        uniq = sorted(set(e["dropped_chars"]))
        report_lines.append(f"{e['stem']}: {len(e['dropped_chars'])} char(s) dropped: {uniq!r}")
        first_line = e['devanagari'].splitlines()[0][:120] if e.get('devanagari') else ''
        report_lines.append(f"  original: {first_line}...")
        report_lines.append("")
    (out_dir / "preprocessing_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8")
    print(f"[stitch] preprocessing report: {out_dir/'preprocessing_report.txt'}")

    # Per-sarga MP3s
    if args.per_sarga:
        by_sarga = defaultdict(list)
        for e in all_entries:
            if e.get("status") in ("rendered", "resumed"):
                by_sarga[e["sarga"]].append(out_dir / e["mp3"])
        for sarga in sorted(by_sarga):
            mp3s = sorted(by_sarga[sarga], key=lambda p: p.name)
            out_mp3 = out_dir / f"sarga-{sarga:02d}.mp3"
            print(f"[stitch] sarga-{sarga:02d}: concatenating {len(mp3s)} shlokas -> {out_mp3.name}")
            concat_mp3(mp3s, out_mp3)

    print("[stitch] done.")


if __name__ == "__main__":
    main()
