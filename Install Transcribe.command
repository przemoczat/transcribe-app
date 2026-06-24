#!/bin/bash
# Double-click this to install Transcribe.
# macOS opens .command files in Terminal, so install.sh runs without typing anything.

# Run from this file's own folder, wherever it was unzipped to.
cd "$(dirname "$0")" || { echo "Could not find the install folder."; exit 1; }

echo "Installing Transcribe — this opens Terminal and may ask for your password (Homebrew)."
echo

/bin/bash ./install.sh
status=$?

echo
if [ "$status" -eq 0 ]; then
  echo "✅ Done. Open  ~/Applications/Transcribe.app"
  echo "   First launch: right-click the app and choose Open (it's not notarized)."
else
  echo "❌ Install stopped (exit $status). Scroll up to see what went wrong."
  echo "   Most common fix: install Homebrew from https://brew.sh, then run this again."
fi
echo
echo "You can close this window."
