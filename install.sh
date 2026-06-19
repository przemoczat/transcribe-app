#!/bin/bash
# Transcribe — installer for the local drag-and-drop transcription app.
# Apple Silicon Mac only. Installs dependencies, the engine scripts and the app.
set -euo pipefail

say() { printf "\033[1m==> %s\033[0m\n" "$1"; }
err() { printf "\033[31m%s\033[0m\n" "$1"; }

# 0. Environment checks ------------------------------------------------------
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  err "Transcribe needs an Apple Silicon Mac (M1/M2/M3/M4). Intel Macs are not supported."
  exit 1
fi
if ! command -v brew >/dev/null 2>&1; then
  err "Homebrew not found. Install it from https://brew.sh and run this script again."
  exit 1
fi

SRC="$(cd "$(dirname "$0")" && pwd)"

# 1. System dependencies -----------------------------------------------------
say "Installing ffmpeg and yt-dlp via Homebrew (skips what you already have)..."
brew install ffmpeg yt-dlp
command -v python3 >/dev/null 2>&1 || brew install python

# 2. Python dependency -------------------------------------------------------
say "Installing mlx-whisper (the local transcription engine)..."
python3 -m pip install --upgrade mlx-whisper --break-system-packages

# 3. Engine scripts ----------------------------------------------------------
DEST="$HOME/skills/transcribe-skill"
say "Copying engine scripts to $DEST ..."
mkdir -p "$DEST"
cp "$SRC/transcribe.py" "$SRC/transcribe_server.py" "$DEST/"

# 4. The app -----------------------------------------------------------------
say "Installing Transcribe.app to ~/Applications ..."
mkdir -p "$HOME/Applications"
rm -rf "$HOME/Applications/Transcribe.app"
cp -R "$SRC/Transcribe.app" "$HOME/Applications/"

# 5. Friendly notes ----------------------------------------------------------
if [[ ! -d "/Applications/Google Chrome.app" ]]; then
  err "Note: Google Chrome was not found. The app opens its window in Chrome — install Chrome to use it."
fi

say "Done."
echo "Open ~/Applications/Transcribe.app (first launch: right-click the app and choose Open)."
echo "Tip: drag it to your Dock. The first transcription downloads the Whisper large-v3 model (~1.5 GB), once."
