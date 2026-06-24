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
from urllib.parse import unquote
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
        # The client percent-encodes the name so non-Latin-1 chars (e.g. Polish
        # ą/ę/ł) survive the HTTP header, which is restricted to ISO-8859-1.
        fname = unquote(self.headers.get("X-Filename", "audio"))
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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500&family=Montserrat:wght@400;500;600&display=swap">
<style>
  /* dabrowskimedia.pl identity — charcoal, one terracotta accent, editorial serif */
  :root { --bg:#141414; --panel:#1a1a1a; --ink:#f2f0ec; --ink-dim:#a8a5a0;
          --ink-faint:#6b6965; --accent:#e85d2f; --err:#e8896f;
          --line:rgba(242,240,236,0.12); --line-strong:rgba(242,240,236,0.25);
          --grid:rgba(242,240,236,0.022);
          --font-display:'Cormorant Garamond',Georgia,serif;
          --font-body:'Montserrat',system-ui,sans-serif;
          --font-mono:ui-monospace,'SF Mono',Menlo,monospace;
          --ease:cubic-bezier(0.16,1,0.3,1); }
  * { box-sizing:border-box; }
  .hidden { display:none; }
  html,body { height:100%; margin:0; }
  body { background-color:var(--bg); color:var(--ink);
         background-image:linear-gradient(var(--grid) 1px,transparent 1px),
                          linear-gradient(90deg,var(--grid) 1px,transparent 1px);
         background-size:60px 60px;
         font:400 14px/1.55 var(--font-body);
         display:flex; flex-direction:column; -webkit-user-select:none; user-select:none; }

  header { padding:20px 22px 18px; border-bottom:1px solid var(--line); flex:0 0 auto; }
  header .eyebrow { font:500 11px/1 var(--font-mono); letter-spacing:.18em;
                    text-transform:uppercase; color:var(--accent); }
  header h1 { margin:6px 0 0; font:500 30px/1 var(--font-display);
              letter-spacing:.01em; color:var(--ink); }
  main { flex:1; min-height:0; overflow-y:auto; padding:22px;
         display:flex; flex-direction:column; gap:20px; }

  #drop { border:1px dashed var(--line-strong); border-radius:2px; padding:30px 22px;
          display:flex; flex-direction:column; align-items:center; justify-content:center;
          gap:10px; text-align:center; color:var(--ink-dim); cursor:pointer; min-height:158px;
          transition:border-color var(--dur,.3s) var(--ease), background var(--dur,.3s) var(--ease),
                     color var(--dur,.3s) var(--ease); }
  #drop:hover, #drop.hover { border-color:var(--accent); color:var(--ink);
          background:rgba(232,93,47,.05); }
  #drop .drop-icon { color:var(--ink-faint); transition:color .3s var(--ease); }
  #drop:hover .drop-icon, #drop.hover .drop-icon { color:var(--accent); }
  #drop .drop-main { font:500 16px/1.2 var(--font-body); color:var(--ink); }
  #drop .sub { font:400 12px/1.5 var(--font-body); color:var(--ink-faint); max-width:34ch; }

  /* hairline "or" divider between the primary drop zone and the link/record inputs */
  .rule { display:flex; align-items:center; gap:14px; color:var(--ink-faint);
          font:500 11px/1 var(--font-mono); letter-spacing:.18em; text-transform:uppercase; }
  .rule::before, .rule::after { content:""; flex:1; height:1px; background:var(--line); }

  .urlrow { display:flex; flex-direction:column; gap:11px; }
  #urls { width:100%; min-height:58px; resize:vertical; background:var(--panel);
          color:var(--ink); border:1px solid var(--line-strong); border-radius:2px; padding:12px 13px;
          font:400 13px/1.55 var(--font-body); -webkit-user-select:text; user-select:text;
          transition:border-color .3s var(--ease); }
  #urls:hover { border-color:var(--ink-dim); }
  #urls:focus { outline:none; border-color:var(--accent); }
  #urls::placeholder { color:var(--ink-faint); }
  .urlrow .actions { display:flex; align-items:center; gap:12px; }
  .urlrow .hint { font:400 11px/1 var(--font-mono); letter-spacing:.04em;
                  color:var(--ink-faint); margin-left:auto; }

  .recrow { display:flex; }
  #rec { display:inline-flex; align-items:center; gap:9px; }
  #rec::before { content:""; width:8px; height:8px; border-radius:50%;
                 background:var(--accent); flex:0 0 auto; }
  #rec.recording { background:var(--accent); border-color:var(--accent); color:var(--bg);
                   font-weight:600; }
  #rec.recording::before { background:var(--bg); animation:pulse 1.1s var(--ease) infinite; }
  @keyframes pulse { 50% { opacity:.2; } }

  #results { display:flex; flex-direction:column; gap:14px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:2px; padding:15px; }
  .card.error { border-color:rgba(232,93,47,.45); }
  .toolbar { display:flex; align-items:center; gap:9px; flex-wrap:wrap; margin-bottom:12px; }
  .toolbar .title { font:500 14px/1.3 var(--font-body); color:var(--ink); margin-right:auto;
                    overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:46%; }
  .working-row { display:flex; align-items:center; gap:9px;
                 font:400 12px/1 var(--font-mono); color:var(--ink-dim); }
  .note { font:400 12px/1.5 var(--font-body); color:var(--ink-dim); }
  .spinner { width:15px; height:15px; border:2px solid var(--line-strong);
             border-top-color:var(--accent); border-radius:50%; animation:spin .8s linear infinite; flex:0 0 auto; }
  @keyframes spin { to { transform:rotate(360deg); } }
  textarea.out { width:100%; height:210px; resize:vertical; background:var(--bg);
             color:var(--ink); border:1px solid var(--line); border-radius:2px; padding:13px;
             font:400 13px/1.65 var(--font-mono);
             -webkit-user-select:text; user-select:text; }
  textarea.out:focus { outline:none; border-color:var(--line-strong); }
  .errmsg { color:var(--err); font:400 12px/1.55 var(--font-mono); word-break:break-word; }
  .erractions { margin-top:12px; }

  button { min-height:38px; background:transparent; color:var(--ink);
           border:1px solid var(--line-strong); border-radius:2px; padding:8px 16px;
           font:500 13px/1 var(--font-body); cursor:pointer;
           transition:color .3s var(--ease), border-color .3s var(--ease), background .3s var(--ease); }
  button:hover { color:var(--accent); border-color:var(--accent); }
  button:active { transform:translateY(1px); }
  button:disabled { opacity:.45; cursor:not-allowed; }
  button.primary { background:var(--accent); border-color:var(--accent); color:var(--bg); font-weight:600; }
  button.primary:hover { background:var(--ink); border-color:var(--ink); color:var(--bg); }
  button.copied, button.copied:hover { background:var(--accent); border-color:var(--accent);
                  color:var(--bg); }
  .switch { display:flex; align-items:center; gap:8px; font:400 12px/1 var(--font-mono);
            color:var(--ink-dim); border:1px solid var(--line-strong); padding:8px 12px;
            border-radius:2px; cursor:pointer; }
  .switch input { accent-color:var(--accent); }

  @media (prefers-reduced-motion: reduce) {
    .spinner { animation:none; }
    #rec.recording::before { animation:none; }
    * { transition:none !important; }
  }
</style>
</head>
<body>
<header>
  <span class="eyebrow">Dabrowski Media</span>
  <h1>Transcribe</h1>
</header>
<main>
  <div id="drop">
    <svg class="drop-icon" width="32" height="32" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="1.25" stroke-linecap="round"
         stroke-linejoin="round" aria-hidden="true">
      <path d="M12 3v11"/><path d="m7.5 10.5 4.5 4.5 4.5-4.5"/><path d="M5 20h14"/>
    </svg>
    <div class="drop-main">Drop audio or video</div>
    <div class="sub">or click to choose — several at once, keep dropping while others run</div>
    <input id="file" type="file" class="hidden" multiple
           accept="audio/*,video/*,.mp3,.m4a,.wav,.mp4,.mov,.aac,.flac,.ogg,.webm,.mkv">
  </div>

  <div class="rule">or</div>

  <div class="urlrow">
    <textarea id="urls" placeholder="…or paste links — YouTube, Instagram, TikTok, direct mp4 — one per line"></textarea>
    <div class="actions">
      <button id="go" class="primary">Transcribe links</button>
      <span class="hint">⌘↵ to transcribe</span>
    </div>
  </div>

  <div class="recrow">
    <button id="rec">● Record voice memo</button>
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
    ? fetch('/transcribe', {method:'POST', body:job.file,
                            headers:{'X-Filename':encodeURIComponent(job.file.name)}})
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

// ---- voice memo recording (reuses the file-transcription path) ----
const recBtn = document.getElementById('rec');
let mediaRecorder = null, recChunks = [], recTimer = null, recT0 = 0;

// Hide the button if the browser can't record (older engines / no mic API).
if (!('MediaRecorder' in window) || !navigator.mediaDevices?.getUserMedia) {
  recBtn.style.display = 'none';
}

function pickMime(){
  const opts = ['audio/webm;codecs=opus','audio/webm','audio/mp4','audio/ogg;codecs=opus'];
  for (const m of opts) if (MediaRecorder.isTypeSupported(m)) return m;
  return '';
}

function recError(text){
  const card = document.createElement('div');
  card.className = 'card';
  results.prepend(card);
  finishCard(card, {ok:false, error:text}, {type:'file', name:'Voice memo'});
}

async function startRecording(){
  recBtn.disabled = true;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({audio:true});
  } catch(e){
    recBtn.disabled = false;
    recError('Microphone access denied — enable it in the browser’s site settings, then try again.');
    return;
  }
  const mime = pickMime();
  mediaRecorder = new MediaRecorder(stream, mime ? {mimeType:mime} : undefined);
  recChunks = [];
  mediaRecorder.ondataavailable = ev => { if (ev.data && ev.data.size) recChunks.push(ev.data); };
  mediaRecorder.onstop = () => {
    stream.getTracks().forEach(t => t.stop());
    const type = mediaRecorder.mimeType || mime || 'audio/webm';
    const ext = type.includes('mp4') ? 'm4a' : type.includes('ogg') ? 'ogg' : 'webm';
    const blob = new Blob(recChunks, {type});
    if (!blob.size) return;  // nothing captured — stay quiet
    const n = new Date(), p = x => String(x).padStart(2,'0');
    const name = `voice-memo-${p(n.getHours())}-${p(n.getMinutes())}-${p(n.getSeconds())}.${ext}`;
    startJob({type:'file', file:new File([blob], name, {type}), name});
  };
  mediaRecorder.start();
  recBtn.disabled = false;
  recBtn.classList.add('recording');
  recT0 = Date.now();
  const tick = () => {
    const s = Math.floor((Date.now()-recT0)/1000);
    recBtn.textContent = 'Stop · ' + Math.floor(s/60)+':'+String(s%60).padStart(2,'0');
  };
  tick();
  recTimer = setInterval(tick, 500);
}

function stopRecording(){
  clearInterval(recTimer);
  recBtn.classList.remove('recording');
  recBtn.textContent = '● Record voice memo';
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
}

recBtn.addEventListener('click', () => {
  if (mediaRecorder && mediaRecorder.state === 'recording') stopRecording();
  else startRecording();
});

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
