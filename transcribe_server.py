#!/usr/bin/env python3
"""
Transcribe — drag-and-drop GUI for the Dabrowski Media transcript engine.

Runs a tiny local web server (stdlib only) and opens it as a frameless
Chrome "app" window. Drag an audio/video file onto the window, the MLX
Whisper engine transcribes it on the GPU, and the text shows up with a
Copy button and a plain / [mm:ss] timestamp toggle.

Reuses the pipeline functions from transcribe.py — no duplicated logic.

Launched by Transcribe.app, but also runnable directly:
    python3 transcribe_server.py
Set TRANSCRIBE_NO_BROWSER=1 to skip opening a window (for testing).
"""

import os
import sys
import json
import time
import threading
import tempfile
import subprocess
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Make the sibling engine importable regardless of cwd.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Apps launched from the Dock/Finder get a minimal PATH that omits Homebrew,
# so ffmpeg/ffprobe/yt-dlp won't be found. Make sure they're on PATH.
for _bindir in ("/opt/homebrew/bin", "/usr/local/bin"):
    if _bindir not in os.environ.get("PATH", "").split(os.pathsep) and os.path.isdir(_bindir):
        os.environ["PATH"] = _bindir + os.pathsep + os.environ.get("PATH", "")

MODEL = "large-v3"          # multilingual; auto-detects language from first 30s
IDLE_TIMEOUT = 30 * 60      # safety net: exit if idle this long with no work

# ─── activity tracking (for the idle watchdog) ──────────────────────────────
_last_activity = time.time()
_active_jobs = 0
_lock = threading.Lock()
# Only one job may use the GPU (Whisper) at a time; downloads/conversion of
# other jobs still overlap freely. Prevents contention / out-of-memory.
_gpu_lock = threading.Lock()


def _touch():
    global _last_activity
    with _lock:
        _last_activity = time.time()


# ─── transcription pipeline (reuses transcribe.py) ──────────────────────────

def _segments_to_text(segments):
    """Turn engine segments into (plain_text, timestamped_text)."""
    import transcribe as eng
    plain_parts, ts_lines = [], []
    for seg in segments:
        text = eng.clean_text(seg["text"])
        if not text:
            continue
        plain_parts.append(text)
        ts_lines.append(f"[{eng.format_ts(seg['start'])}]  {text}")
    return " ".join(plain_parts), "\n".join(ts_lines)


def transcribe_file(path: str):
    """Return (plain, timestamped, title) for a local audio/video file."""
    import transcribe as eng
    with tempfile.TemporaryDirectory() as tmp:
        wav = eng.convert_audio(path, tmp, "")
        chunks = eng.split_audio(wav, tmp, "")
        with _gpu_lock:
            segments = eng.transcribe_chunks(chunks, MODEL, None, "")
    plain, ts = _segments_to_text(eng.deduplicate(segments))
    return plain, ts, ""


def transcribe_url(url: str):
    """Return (plain, timestamped, title) for an internet link.

    Uses the platform's captions when available (fast); otherwise downloads
    the audio and runs large-v3. Mirrors the CLI's run_video logic.
    """
    import transcribe as eng
    with tempfile.TemporaryDirectory() as tmp:
        segments, title = eng.try_download_captions(url, tmp, "", None)
        if segments is None:
            audio_file, title = eng.download_audio(url, tmp, "")
            wav = eng.convert_audio(audio_file, tmp, "")
            chunks = eng.split_audio(wav, tmp, "")
            with _gpu_lock:
                segments = eng.transcribe_chunks(chunks, MODEL, None, "")
    plain, ts = _segments_to_text(eng.deduplicate(segments))
    return plain, ts, (title or url)


# ─── HTTP handler ───────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # quiet

    def _send(self, code, body: bytes, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _read_body(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.rfile.read(min(1 << 16, n - len(buf)))
            if not chunk:
                break
            buf.extend(chunk)
        return bytes(buf)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        _touch()

        if self.path == "/shutdown":
            self._send(200, b"bye", "text/plain")
            threading.Thread(target=httpd.shutdown, daemon=True).start()
            return

        if self.path == "/transcribe":
            self._run_job(self._do_file)
        elif self.path == "/transcribe_url":
            self._run_job(self._do_url)
        elif self.path == "/update_ytdlp":
            self._do_update()
        else:
            self._send(404, b"not found", "text/plain")

    def _do_update(self):
        """Upgrade yt-dlp (brew if available, else pip). Returns {ok, output}."""
        import shutil as sh
        if sh.which("brew"):
            cmd = ["brew", "upgrade", "yt-dlp"]
        else:
            cmd = [sys.executable, "-m", "pip", "install", "-U",
                   "yt-dlp", "--break-system-packages"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
            ok = proc.returncode == 0
            out = (proc.stdout + proc.stderr).strip()
        except Exception as e:
            ok, out = False, str(e)
        payload = {"ok": ok, "output": out[-1500:] or "Done."}
        self._send(200, json.dumps(payload).encode("utf-8"), "application/json")

    def _run_job(self, job):
        """Run a transcription job, tracking activity and catching errors."""
        global _active_jobs
        with _lock:
            _active_jobs += 1
        try:
            plain, ts, title, fallback_title = job()
            payload = {"ok": True, "plain": plain, "timestamped": ts,
                       "title": title or fallback_title}
        except Exception as e:  # surface the error in the UI card
            payload = {"ok": False, "error": str(e)}
        finally:
            with _lock:
                _active_jobs -= 1
            _touch()
        self._send(200, json.dumps(payload).encode("utf-8"), "application/json")

    def _do_file(self):
        length = int(self.headers.get("Content-Length", "0"))
        fname = self.headers.get("X-Filename", "audio")
        suffix = os.path.splitext(fname)[1] or ".bin"
        data = self._read_body(length)
        tmpf = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmpf.write(data)
        tmpf.close()
        try:
            plain, ts, title = transcribe_file(tmpf.name)
        finally:
            try:
                os.unlink(tmpf.name)
            except OSError:
                pass
        return plain, ts, title, os.path.splitext(fname)[0]

    def _do_url(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self._read_body(length) or b"{}")
        url = (body.get("url") or "").strip()
        if not url:
            raise ValueError("No URL provided")
        plain, ts, title = transcribe_url(url)
        return plain, ts, title, url


# ─── browser launch ─────────────────────────────────────────────────────────

def launch_browser(url):
    profile = Path.home() / "Library/Application Support/TranscribeApp/chrome-profile"
    profile.mkdir(parents=True, exist_ok=True)
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    ]
    binary = next((c for c in candidates if os.path.exists(c)), None)
    if binary:
        subprocess.Popen([
            binary,
            f"--app={url}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--window-size=600,760",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        import webbrowser
        webbrowser.open(url)


# ─── idle watchdog ──────────────────────────────────────────────────────────

def watchdog():
    while True:
        time.sleep(60)
        with _lock:
            idle = time.time() - _last_activity
            busy = _active_jobs > 0
        if not busy and idle > IDLE_TIMEOUT:
            httpd.shutdown()
            return


# ─── HTML (single inline page) ──────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Transcribe</title>
<style>
  :root { --bg:#0f1115; --panel:#181b22; --line:#262b35; --accent:#4f8cff;
          --text:#e7ebf2; --muted:#8b93a3; --ok:#3ecf8e; --err:#ff7676; }
  * { box-sizing:border-box; }
  html,body { height:100%; margin:0; }
  body { background:var(--bg); color:var(--text);
         font:14px/1.5 -apple-system,BlinkMacSystemFont,"SF Pro Text",sans-serif;
         display:flex; flex-direction:column; -webkit-user-select:none; user-select:none; }
  header { padding:13px 18px; border-bottom:1px solid var(--line);
           display:flex; align-items:center; gap:10px; flex:0 0 auto; }
  header h1 { font-size:15px; margin:0; font-weight:600; letter-spacing:.2px; }
  header .dot { width:9px; height:9px; border-radius:50%; background:var(--ok); }
  main { flex:1; min-height:0; overflow-y:auto; padding:16px; display:flex; flex-direction:column; gap:14px; }

  #drop { border:2px dashed var(--line); border-radius:14px; padding:22px;
          display:flex; flex-direction:column; align-items:center; justify-content:center;
          gap:7px; text-align:center; color:var(--muted); cursor:pointer; min-height:140px;
          transition:border-color .15s, background .15s; }
  #drop.hover { border-color:var(--accent); background:rgba(79,140,255,.07); color:var(--text); }
  #drop .big { font-size:30px; opacity:.85; }
  #drop .sub { font-size:12px; }

  .urlrow { display:flex; flex-direction:column; gap:8px; }
  #urls { width:100%; min-height:56px; resize:vertical; background:var(--panel);
          color:var(--text); border:1px solid var(--line); border-radius:11px; padding:11px 12px;
          font:13px/1.5 -apple-system,sans-serif; -webkit-user-select:text; user-select:text; }
  #urls::placeholder { color:var(--muted); }
  .urlrow .actions { display:flex; align-items:center; gap:10px; }
  .urlrow .hint { color:var(--muted); font-size:11px; margin-left:auto; }

  #results { display:flex; flex-direction:column; gap:12px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px; }
  .card.error { border-color:rgba(255,118,118,.4); }
  .toolbar { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:9px; }
  .toolbar .title { font-weight:600; margin-right:auto; overflow:hidden;
                    text-overflow:ellipsis; white-space:nowrap; max-width:46%; }
  .working-row { display:flex; align-items:center; gap:8px; color:var(--muted); font-size:12px; }
  .note { color:var(--muted); font-size:12px; }
  .spinner { width:16px; height:16px; border:3px solid var(--line);
             border-top-color:var(--accent); border-radius:50%; animation:spin .8s linear infinite; flex:0 0 auto; }
  @keyframes spin { to { transform:rotate(360deg); } }
  textarea.out { width:100%; height:200px; resize:vertical; background:var(--bg);
             color:var(--text); border:1px solid var(--line); border-radius:10px; padding:12px;
             font:13px/1.6 "SF Mono",ui-monospace,Menlo,monospace;
             -webkit-user-select:text; user-select:text; }
  .errmsg { color:var(--err); font:12px/1.5 "SF Mono",ui-monospace,monospace; word-break:break-word; }
  .erractions { margin-top:10px; }
  button { background:var(--panel); color:var(--text); border:1px solid var(--line);
           padding:8px 14px; border-radius:9px; font-size:13px; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  button:disabled { opacity:.6; cursor:default; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
  button.copied { background:var(--ok); border-color:var(--ok); color:#06251a; }
  .switch { display:flex; align-items:center; gap:7px; color:var(--muted); font-size:12px;
            border:1px solid var(--line); padding:7px 11px; border-radius:9px; cursor:pointer; }
  .switch input { accent-color:var(--accent); }
</style>
</head>
<body>
<header><span class="dot"></span><h1>Transcribe</h1></header>
<main>
  <div id="drop">
    <div class="big">⤓</div>
    <div>Drop audio or video files here</div>
    <div class="sub">or click to choose · multiple at once · keep dropping while others run</div>
    <input id="file" type="file" class="hidden" multiple
           accept="audio/*,video/*,.mp3,.m4a,.wav,.mp4,.mov,.aac,.flac,.ogg,.webm,.mkv">
  </div>

  <div class="urlrow">
    <textarea id="urls" placeholder="…or paste links — YouTube, Instagram, TikTok, direct mp4 — one per line"></textarea>
    <div class="actions">
      <button id="go" class="primary">Transcribe links</button>
      <span class="hint">⌘↵ to transcribe</span>
    </div>
  </div>

  <div id="results"></div>
</main>

<script>
const drop = document.getElementById('drop');
const fileInput = document.getElementById('file');
const urlsBox = document.getElementById('urls');
const results = document.getElementById('results');

// ---- drag & drop (anywhere in the window) ----
['dragenter','dragover'].forEach(e =>
  drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add('hover'); }));
['dragleave','drop'].forEach(e =>
  drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove('hover'); }));
window.addEventListener('dragover', ev => ev.preventDefault());
window.addEventListener('drop', ev => ev.preventDefault());

drop.addEventListener('drop', ev => {
  [...ev.dataTransfer.files].forEach(f => startJob({type:'file', file:f, name:f.name}));
});
drop.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  [...fileInput.files].forEach(f => startJob({type:'file', file:f, name:f.name}));
  fileInput.value = '';
});

// ---- url submit ----
function parseUrls(text){
  return text.split(/\s+/).map(s => s.trim()).filter(s => /^https?:\/\//i.test(s));
}
function submitUrls(){
  const urls = parseUrls(urlsBox.value);
  if (!urls.length) return;
  urls.forEach(u => startJob({type:'url', url:u, name:u}));
  urlsBox.value = '';
}
document.getElementById('go').addEventListener('click', submitUrls);
urlsBox.addEventListener('keydown', e => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); submitUrls(); }
});

// ---- jobs: each starts immediately, shows its own card + timer ----
function startJob(job){
  const card = document.createElement('div');
  card.className = 'card working';
  card.innerHTML =
    '<div class="toolbar">' +
      '<span class="title"></span>' +
      '<span class="working-row"><span class="spinner"></span><span class="elapsed">0:00</span></span>' +
    '</div>' +
    '<div class="note">Transcribing…</div>';
  card.querySelector('.title').textContent = job.name;
  results.prepend(card);

  const t0 = Date.now();
  const elapsed = card.querySelector('.elapsed');
  card._timer = setInterval(() => {
    const s = Math.floor((Date.now()-t0)/1000);
    elapsed.textContent = Math.floor(s/60)+':'+String(s%60).padStart(2,'0');
  }, 500);

  const req = job.type === 'file'
    ? fetch('/transcribe', {method:'POST', body:job.file, headers:{'X-Filename':job.file.name}})
    : fetch('/transcribe_url', {method:'POST', body:JSON.stringify({url:job.url}),
                                headers:{'Content-Type':'application/json'}});

  req.then(r => r.json()).catch(e => ({ok:false, error:String(e)}))
     .then(res => finishCard(card, res, job));
}

function finishCard(card, res, job){
  clearInterval(card._timer);
  card.classList.remove('working');
  card.innerHTML = '';
  if (res && res.ok) renderResult(card, res, job);
  else renderError(card, res || {}, job);
}

function renderResult(card, res, job){
  card.innerHTML =
    '<div class="toolbar">' +
      '<span class="title"></span>' +
      '<label class="switch"><input type="checkbox" class="ts"> Timestamps</label>' +
      '<button class="copy primary">Copy</button>' +
      '<button class="save">Save .txt…</button>' +
    '</div>' +
    '<textarea class="out" readonly spellcheck="false"></textarea>';
  const title = res.title || job.name || 'Transcript';
  card.querySelector('.title').textContent = title;
  const ta  = card.querySelector('.out');
  const tsb = card.querySelector('.ts');
  const cur = () => tsb.checked ? res.timestamped : res.plain;
  const draw = () => { ta.value = cur(); };
  draw();
  tsb.addEventListener('change', draw);

  const cp = card.querySelector('.copy');
  cp.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(cur()); }
    catch { ta.select(); document.execCommand('copy'); }
    cp.textContent = 'Copied ✓'; cp.classList.add('copied');
    setTimeout(() => { cp.textContent = 'Copy'; cp.classList.remove('copied'); }, 1400);
  });
  card.querySelector('.save').addEventListener('click', () => {
    const blob = new Blob([cur()], {type:'text/plain'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = title.replace(/[\/\\:]/g,'_').slice(0,80) + '.txt';
    a.click();
    URL.revokeObjectURL(a.href);
  });
}

function renderError(card, res, job){
  card.classList.add('error');
  const bar = document.createElement('div'); bar.className = 'toolbar';
  const title = document.createElement('span'); title.className = 'title';
  title.textContent = '⚠ ' + (job.name || 'Error');
  bar.appendChild(title); card.appendChild(bar);

  const msg = document.createElement('div'); msg.className = 'errmsg';
  msg.textContent = res.error || 'Failed';
  card.appendChild(msg);

  if (job.type === 'url'){
    const row = document.createElement('div'); row.className = 'erractions';
    const btn = document.createElement('button');
    btn.textContent = 'Update yt-dlp & retry';
    btn.addEventListener('click', () => updateAndRetry(card, job, btn, msg));
    row.appendChild(btn); card.appendChild(row);
  }
}

async function updateAndRetry(card, job, btn, msg){
  btn.disabled = true; btn.textContent = 'Updating yt-dlp…';
  let res;
  try { res = await fetch('/update_ytdlp', {method:'POST'}).then(r => r.json()); }
  catch(e){ res = {ok:false, output:String(e)}; }
  if (res.ok){ card.remove(); startJob(job); }
  else {
    msg.textContent = 'Update failed: ' + (res.output || 'unknown error');
    btn.disabled = false; btn.textContent = 'Update yt-dlp & retry';
  }
}

// ---- tell the server to exit when the window closes ----
window.addEventListener('pagehide', () => { navigator.sendBeacon('/shutdown'); });
</script>
</body>
</html>
"""


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    global httpd
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    url = f"http://127.0.0.1:{port}/"

    print(f"Transcribe server running at {url}", flush=True)

    threading.Thread(target=watchdog, daemon=True).start()

    if os.environ.get("TRANSCRIBE_NO_BROWSER") != "1":
        launch_browser(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    print("Transcribe server stopped.", flush=True)


if __name__ == "__main__":
    main()
