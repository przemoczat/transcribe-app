# Transcribe

A tiny drag-and-drop transcription app for macOS. Drop an audio or video file (or paste a
link), get clean text with timestamps. Everything runs **locally** on your Mac — no API key,
no cost, nothing leaves your machine.

Built by [Przemek Dąbrowski](https://dabrowskimedia.pl) for everyday content work, shared in
case it's useful to you too.

## What it does

- Transcribes **local audio and video** files (ffmpeg strips the audio automatically).
- Transcribes from **links**: YouTube, Instagram, TikTok, Vimeo, a direct MP4 — anything
  yt-dlp supports. Uses existing captions when available, otherwise downloads and transcribes.
- **Record a voice memo** right in the app — click record, talk, stop, get the transcript.
- Output as plain text or with `[mm:ss]` timestamps, one click to copy or save a `.txt`.
- Drop several files at once — each runs as its own card with a live timer.
- Auto-detects the language (Polish, English, and many more). Powered by Whisper large-v3.
- Roughly **16× realtime** on Apple Silicon (a 4-minute clip ≈ 15 seconds).

## Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4). Intel Macs are not supported — it uses Apple's MLX.
- **Homebrew** ([brew.sh](https://brew.sh)).
- **Google Chrome** (the app opens its window as a Chrome app window).

## Install

```bash
git clone https://github.com/przemoczat/transcribe-app.git
cd transcribe-app
./install.sh
```

The installer adds `ffmpeg`, `yt-dlp` and `mlx-whisper`, copies the engine scripts to
`~/skills/transcribe-skill/` and installs `Transcribe.app` to `~/Applications/`.

First launch: right-click the app and choose **Open** (it's not notarized, so macOS asks once).
Then drag it to your Dock. The first transcription downloads the Whisper large-v3 model
(~1.5 GB), once.

## How it works

`Transcribe.app` is a small launcher that starts a local server
(`transcribe_server.py`) on a random localhost port and opens a frameless Chrome window
pointing at it. The server wraps `transcribe.py` (MLX Whisper large-v3). The GPU step is
serialized so several queued files don't fight over memory. Closing the window shuts the
server down; there's also a 30-minute idle watchdog.

## Privacy

Files are processed locally with MLX Whisper. No audio, video or text is uploaded anywhere.
Link downloads go through yt-dlp directly from your machine.

## License

MIT — see [LICENSE](LICENSE).
