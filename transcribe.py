#!/usr/bin/env python3
"""
Transcript Engine — Dabrowski Media
Przemek / Claude Code Skill

Uses mlx-whisper (Apple Silicon GPU) — 10–20x faster than CPU.
Supports multiple URLs in parallel — each gets its own TXT file.
Also accepts local file paths (MP4, MOV, MP3, WAV, M4A, etc.).

Usage:
  python3 transcribe.py <URL>
  python3 transcribe.py /path/to/video.mp4        # local file
  python3 transcribe.py <URL1> <URL2> <URL3>      # parallel
  python3 transcribe.py <URL> --lang pl
  python3 transcribe.py <URL> --model large-v3    # higher quality
"""

import os
import sys
import re
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


# ─── CONFIG ──────────────────────────────────────────────────────────────────

DOWNLOADS_DIR = Path.home() / "Downloads"
CHUNK_MINUTES = 25
DEFAULT_MODEL = "large-v3"   # multilingual — handles Polish, English, etc.

# mlx-community HuggingFace model IDs
MLX_MODELS = {
    "tiny":           "mlx-community/whisper-tiny-mlx",
    "base":           "mlx-community/whisper-base-mlx",
    "small":          "mlx-community/whisper-small-mlx",
    "medium":         "mlx-community/whisper-medium-mlx",
    "large-v2":       "mlx-community/whisper-large-v2-mlx",
    "large-v3":       "mlx-community/whisper-large-v3-mlx",
    "distil-large-v3":"mlx-community/distil-whisper-large-v3",  # English only
}

# Models that only support English — need auto-switch for other languages
ENGLISH_ONLY_MODELS = {"distil-large-v3"}


# ─── DEPENDENCY CHECK ────────────────────────────────────────────────────────

def check_deps():
    missing = [cmd for cmd in ["yt-dlp", "ffmpeg"] if shutil.which(cmd) is None]
    if missing:
        print(f"[ERROR] Missing: {', '.join(missing)}")
        print("  Install: brew install " + " ".join(missing))
        sys.exit(1)
    try:
        import mlx_whisper  # noqa: F401
    except ImportError:
        print("[ERROR] mlx-whisper not installed.")
        print("  Install: pip install mlx-whisper --break-system-packages")
        sys.exit(1)


# ─── LOGGING ─────────────────────────────────────────────────────────────────

def log(prefix, msg):
    print(f"{prefix}{msg}", flush=True)


# ─── DOWNLOAD ────────────────────────────────────────────────────────────────

def download_audio(url, tmpdir, prefix):
    log(prefix, "Downloading audio...")
    out_template = os.path.join(tmpdir, "audio.%(ext)s")
    cmd = [
        "yt-dlp", "--no-playlist", "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--no-warnings",
        "--print", "%(title)s",
        "--no-simulate",
        "-o", out_template,
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Download failed:\n{result.stderr.strip()}")

    title = result.stdout.strip().split("\n")[0]
    matches = list(Path(tmpdir).glob("audio.*"))
    if not matches:
        raise RuntimeError("No audio file found after download.")

    log(prefix, f"  Got: {title or matches[0].name}")
    return str(matches[0]), title


# ─── CONVERT ─────────────────────────────────────────────────────────────────

def convert_audio(input_path, tmpdir, prefix):
    log(prefix, "Converting to WAV (16kHz mono)...")
    output_path = os.path.join(tmpdir, "audio.wav")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        "-loglevel", "error",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.strip()}")
    return output_path


# ─── DURATION ────────────────────────────────────────────────────────────────

def get_duration(path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


# ─── CHUNK SPLITTER ───────────────────────────────────────────────────────────

def split_audio(audio_path, tmpdir, prefix):
    duration  = get_duration(audio_path)
    chunk_sec = CHUNK_MINUTES * 60

    if duration <= chunk_sec:
        return [(audio_path, 0.0)]

    chunks, start, idx = [], 0.0, 0
    while start < duration:
        chunk_path = os.path.join(tmpdir, f"chunk_{idx:03d}.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(start), "-t", str(chunk_sec),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            "-loglevel", "error", chunk_path,
        ], capture_output=True)
        chunks.append((chunk_path, start))
        start += chunk_sec
        idx   += 1

    log(prefix, f"  Split into {len(chunks)} × {CHUNK_MINUTES}-min chunks")
    return chunks


# ─── CAPTIONS ────────────────────────────────────────────────────────────────

def try_download_captions(url, tmpdir, prefix, language):
    """
    Try to fetch captions/subtitles directly from the platform (instant).
    Returns (segments, title) if found, (None, title) otherwise.
    Checks manual subs first, then auto-generated.
    """
    log(prefix, "Checking for captions...")

    sub_langs = f"{language},-live_chat" if language else "all,-live_chat"
    out_tmpl  = os.path.join(tmpdir, "%(title)s")

    cmd = [
        "yt-dlp", "--no-playlist",
        "--write-subs", "--write-auto-subs",
        "--sub-langs", sub_langs,
        "--convert-subs", "srt",
        "--skip-download",
        "--no-warnings",
        "--print", "%(title)s",
        "--no-simulate",
        "-o", out_tmpl,
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    title = result.stdout.strip().split("\n")[0] if result.stdout.strip() else ""

    srt_files = sorted(Path(tmpdir).glob("*.srt"))
    if not srt_files:
        log(prefix, "  No captions available — will transcribe audio")
        return None, title

    # Prefer a file matching the requested language; otherwise first available
    chosen = srt_files[0]
    if language:
        matches = [f for f in srt_files if f".{language}." in f.name]
        if matches:
            chosen = matches[0]

    log(prefix, f"  Captions found: {chosen.name} — skipping audio transcription!")
    segments = parse_srt(chosen)
    return segments, title


def parse_srt(srt_path):
    """Parse SRT file into [{start, text}, ...] segment list."""
    text   = Path(srt_path).read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n{2,}", text.strip())
    segments = []

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        ts_match = re.match(r"(\d{2}):(\d{2}):(\d{2}),\d+", lines[1])
        if not ts_match:
            continue
        h, m, s = int(ts_match.group(1)), int(ts_match.group(2)), int(ts_match.group(3))
        raw = " ".join(lines[2:])
        # Strip HTML/XML tags (YouTube uses <c>, <b>, <timestamp> etc.)
        clean = re.sub(r"<[^>]+>", "", raw).strip()
        if clean:
            segments.append({"start": float(h * 3600 + m * 60 + s), "text": clean})

    return segments


# ─── LANGUAGE DETECTION ──────────────────────────────────────────────────────

def detect_language(wav_path, hf_repo, prefix):
    """Detect language from first 30s of audio. Returns ISO code e.g. 'pl', 'en'."""
    import mlx_whisper

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sample_path = f.name

    subprocess.run([
        "ffmpeg", "-y", "-i", wav_path,
        "-t", "30",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        "-loglevel", "error", sample_path,
    ], capture_output=True)

    result = mlx_whisper.transcribe(sample_path, path_or_hf_repo=hf_repo)
    os.unlink(sample_path)

    lang = result.get("language", "en")
    log(prefix, f"  Detected language: {lang}")
    return lang


# ─── TRANSCRIBE (MLX — Apple Silicon GPU) ────────────────────────────────────

def transcribe_chunks(chunks, model_size, language, prefix):
    import mlx_whisper

    hf_repo = MLX_MODELS[model_size]
    log(prefix, f"Loading model '{model_size}' via MLX (GPU)...")

    # Auto-detect language from first chunk if not specified
    if not language:
        language = detect_language(chunks[0][0], hf_repo, prefix)

    # If English-only model is selected but content isn't English, upgrade model
    if model_size in ENGLISH_ONLY_MODELS and language != "en":
        model_size = "large-v3"
        hf_repo    = MLX_MODELS[model_size]
        log(prefix, f"  Switching to '{model_size}' (multilingual) for language: {language}")

    all_segments = []

    for i, (chunk_path, time_offset) in enumerate(chunks, 1):
        log(prefix, f"Transcribing chunk {i}/{len(chunks)} [{language}]...")

        result = mlx_whisper.transcribe(
            chunk_path,
            path_or_hf_repo=hf_repo,
            language=language,
            word_timestamps=False,
            condition_on_previous_text=False,  # prevents hallucination loops
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
        )

        for seg in result.get("segments", []):
            all_segments.append({
                "start": seg["start"] + time_offset,
                "text":  seg["text"].strip(),
            })

        log(prefix, f"  Chunk {i}/{len(chunks)} done — {len(result.get('segments', []))} segments")

    return all_segments


# ─── CLEANUP + OUTPUT ─────────────────────────────────────────────────────────

def deduplicate(segments, window=4):
    """Remove consecutive repeated segments — Whisper hallucination artifact."""
    result = []
    for seg in segments:
        recent = [s["text"] for s in result[-window:]]
        if seg["text"] not in recent:
            result.append(seg)
    return result


def clean_text(text):
    for pattern in [r'\b(uh+|um+|hmm+|mhm+|erm+|uhh+)\b', r'\b(eee+|aaa+)\b', r'\[.*?\]']:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'\s([,\.!?;:])', r'\1', text)
    text = re.sub(r'(?<=[.!?])\s+([a-z])', lambda m: m.group(0).upper(), text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text.strip()


def format_ts(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def build_txt(segments, url):
    lines = ["TRANSCRIPT", f"Source: {url}", "=" * 60, ""]
    for seg in segments:
        text = clean_text(seg["text"])
        if text:
            lines.append(f"[{format_ts(seg['start'])}]  {text}")
    return "\n".join(lines)


def make_filename(title):
    if title:
        name = re.sub(r'[^\w\s-]', '', title)
        name = re.sub(r'\s+', '_', name.strip())
        return f"{name[:80]}_transcript.txt"
    return "transcript.txt"


# ─── LOCAL FILE DETECTION ─────────────────────────────────────────────────────

def is_local_file(path):
    p = Path(path).expanduser()
    return p.exists() and p.is_file()


# ─── SINGLE-VIDEO PIPELINE ────────────────────────────────────────────────────

def run_video(url, model_size, language, prefix=""):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            if is_local_file(url):
                local_path = Path(url).expanduser()
                title = local_path.stem
                log(prefix, f"Local file: {local_path.name}")
                wav_file = convert_audio(str(local_path), tmpdir, prefix)
                chunks   = split_audio(wav_file, tmpdir, prefix)
                segments = transcribe_chunks(chunks, model_size, language, prefix)
            else:
                # Fast path: grab captions if available
                segments, title = try_download_captions(url, tmpdir, prefix, language)

                if segments is None:
                    # Slow path: download audio and transcribe
                    audio_file, title = download_audio(url, tmpdir, prefix)
                    wav_file          = convert_audio(audio_file, tmpdir, prefix)
                    chunks            = split_audio(wav_file, tmpdir, prefix)
                    segments          = transcribe_chunks(chunks, model_size, language, prefix)

        segments = deduplicate(segments)
        txt      = build_txt(segments, url)
        filename = make_filename(title)
        out_path = DOWNLOADS_DIR / filename
        out_path.write_text(txt, encoding="utf-8")

        log(prefix, f"✓ Saved → {out_path}  ({len(segments)} segments)")
        return str(out_path), None
    except Exception as e:
        log(prefix, f"✗ FAILED: {e}")
        return None, str(e)


def _worker(args):
    url, model_size, language, prefix = args
    return run_video(url, model_size, language, prefix)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Transcript Engine")
    parser.add_argument("urls", nargs="+", help="One or more video URLs or local file paths")
    parser.add_argument("--lang",  default=None, help="Language hint: pl, en, de, …")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MLX_MODELS.keys()),
                        help=f"Whisper model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    check_deps()

    n = len(args.urls)
    print(f"\n Transcript Engine  |  model={args.model} [MLX/GPU]  |  {n} video(s)\n")

    if n == 1:
        run_video(args.urls[0], args.model, args.lang)
    else:
        tasks = [
            (url, args.model, args.lang, f"[{i}/{n}] ")
            for i, url in enumerate(args.urls, 1)
        ]
        with ProcessPoolExecutor(max_workers=n) as pool:
            futures = {pool.submit(_worker, t): t[0] for t in tasks}
            for future in as_completed(futures):
                _, err = future.result()
                if err:
                    print(f"ERROR for {futures[future]}: {err}", flush=True)

    print("\n All done.")


if __name__ == "__main__":
    main()
