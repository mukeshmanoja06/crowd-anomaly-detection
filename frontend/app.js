/* ============================================================
   SENTRY console — client logic
   Handles: file upload, websocket streaming, live telemetry,
   alert ticker, calibration progress, clock.
   ============================================================ */

const els = {
  fileDrop: document.getElementById('fileDrop'),
  fileInput: document.getElementById('fileInput'),
  fileDropText: document.getElementById('fileDropText'),
  liveSource: document.getElementById('liveSource'),
  modelSelect: document.getElementById('modelSelect'),
  confSlider: document.getElementById('confSlider'),
  confValue: document.getElementById('confValue'),
  startBtn: document.getElementById('startBtn'),
  stopBtn: document.getElementById('stopBtn'),
  videoFrame: document.getElementById('videoFrame'),
  cctvPreview: document.getElementById('cctvPreview'),
  cameraPreview: document.getElementById('cameraPreview'),
  videoEmpty: document.getElementById('videoEmpty'),
  calibOverlay: document.getElementById('calibOverlay'),
  calibFill: document.getElementById('calibFill'),
  calibText: document.getElementById('calibText'),
  peopleCount: document.getElementById('peopleCount'),
  fpsValue: document.getElementById('fpsValue'),
  meterMag: document.getElementById('meterMag'),
  meterStd: document.getElementById('meterStd'),
  meterEnt: document.getElementById('meterEnt'),
  valMag: document.getElementById('valMag'),
  valStd: document.getElementById('valStd'),
  valEnt: document.getElementById('valEnt'),
  alertTicker: document.getElementById('alertTicker'),
  connStatus: document.getElementById('connStatus'),
  clock: document.getElementById('clock'),
  sessionLabel: document.getElementById('sessionLabel'),
};

let ws = null;
let uploadedSourcePath = null;
let calibTotalFrames = 60; // mirrors backend default; cosmetic only
let demoTimer = null;
let cameraStream = null;
const apiBase = (window.SENTRY_API_URL || window.location.origin).replace(/\/$/, '');

function apiUrl(path) {
  return `${apiBase}${path}`;
}

function websocketUrl(path) {
  const url = new URL(apiUrl(path));
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.toString();
}

// ---------------- Clock ----------------
function tickClock() {
  const now = new Date();
  els.clock.textContent = now.toLocaleTimeString('en-GB');
}
setInterval(tickClock, 1000);
tickClock();

// ---------------- Confidence slider ----------------
els.confSlider.addEventListener('input', () => {
  els.confValue.textContent = parseFloat(els.confSlider.value).toFixed(2);
});

// ---------------- File upload ----------------
els.fileDrop.addEventListener('click', () => els.fileInput.click());
els.fileDrop.addEventListener('dragover', (e) => {
  e.preventDefault();
  els.fileDrop.classList.add('dragover');
});
els.fileDrop.addEventListener('dragleave', () => els.fileDrop.classList.remove('dragover'));
els.fileDrop.addEventListener('drop', (e) => {
  e.preventDefault();
  els.fileDrop.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
els.fileInput.addEventListener('change', () => {
  if (els.fileInput.files.length) handleFile(els.fileInput.files[0]);
});

async function handleFile(file) {
  els.fileDropText.textContent = `Uploading "${file.name}"…`;
  const form = new FormData();
  form.append('file', file);
  try {
    const res = await fetch(apiUrl('/api/upload'), { method: 'POST', body: form });
    if (!res.ok) throw new Error(`Upload failed (${res.status})`);
    const data = await res.json();
    uploadedSourcePath = data.source;
    els.fileDropText.textContent = `✓ ${data.filename} ready`;
    els.liveSource.value = ''; // uploaded file takes priority
  } catch (err) {
    uploadedSourcePath = URL.createObjectURL(file);
    els.fileDropText.textContent = `${file.name} ready for browser preview`;
    console.error(err);
  }
}

// ---------------- Start / Stop ----------------
els.startBtn.addEventListener('click', startMonitoring);
els.stopBtn.addEventListener('click', stopMonitoring);

function startMonitoring() {
  const liveVal = els.liveSource.value.trim();
  const source = liveVal !== '' ? liveVal : (uploadedSourcePath || '0');

  resetUI();
  setConnState('connecting');
  els.startBtn.disabled = true;

  ws = new WebSocket(websocketUrl('/ws/stream'));

  ws.onopen = () => {
    ws.send(JSON.stringify({
      source: source,
      model: els.modelSelect.value,
      conf: parseFloat(els.confSlider.value),
    }));
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    handleMessage(msg);
  };

  ws.onerror = () => {
    if (window.SENTRY_DEMO_MODE !== false) startDemo(liveVal);
    else {
      setConnState('idle');
      els.startBtn.disabled = false;
    }
  };

  ws.onclose = () => {
    if (demoTimer) return;
    setConnState('idle');
    els.startBtn.disabled = false;
    els.stopBtn.disabled = true;
  };
}

function stopMonitoring() {
  stopDemo();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: 'stop' }));
    ws.close();
  }
  setConnState('idle');
  els.startBtn.disabled = false;
  els.stopBtn.disabled = true;
}

function startDemo(source = '') {
  stopDemo();
  setConnState('demo');
  els.startBtn.disabled = true;
  els.stopBtn.disabled = false;
  els.videoEmpty.style.display = 'none';
  els.videoFrame.style.display = 'none';
  startBrowserSource(source);
  els.sessionLabel.textContent = 'Browser demo mode';
  let frame = 0;
  demoTimer = setInterval(() => {
    frame += 1;
    const wave = (Math.sin(frame / 10) + 1) / 2;
    updateTelemetry({
      people_count: Math.round(8 + wave * 5),
      fps: 30,
      mean_magnitude: 0.8 + wave * 2.2,
      motion_std: 0.5 + wave * 1.8,
      direction_entropy: 0.25 + wave * 0.55,
      calibrating: frame <= calibTotalFrames,
      frame_idx: frame,
      anomalies: frame % 45 === 0 ? [{
        type: 'UNUSUAL_MOTION', label: 'Demo motion event', score: 0.72,
        detail: 'browser-only demonstration',
      }] : [],
    });
  }, 1000 / 30);
}

function stopDemo() {
  if (demoTimer) clearInterval(demoTimer);
  demoTimer = null;
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }
  els.cameraPreview.srcObject = null;
  els.cameraPreview.style.display = 'none';
  els.cctvPreview.removeAttribute('src');
  els.cctvPreview.style.display = 'none';
  els.startBtn.disabled = false;
  els.stopBtn.disabled = true;
}

async function startCameraPreview() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    els.cameraPreview.srcObject = cameraStream;
    els.cameraPreview.style.display = 'block';
  } catch (err) {
    els.sessionLabel.textContent = 'Demo mode · camera permission needed';
    console.warn('Camera preview unavailable:', err);
  }
}

function startBrowserSource(source) {
  const trimmedSource = (source || '').trim();
  if (/^https?:\/\//i.test(trimmedSource)) {
    els.cctvPreview.src = trimmedSource;
    els.cctvPreview.style.display = 'block';
    els.sessionLabel.textContent = 'Browser demo · CCTV link';
    els.cctvPreview.onerror = () => {
      els.sessionLabel.textContent = 'CCTV link blocked or unavailable';
    };
    return;
  }
  if (/^rtsp:\/\//i.test(trimmedSource)) {
    els.sessionLabel.textContent = 'RTSP needs the Python backend';
    return;
  }
  startCameraPreview();
}

function handleMessage(msg) {
  if (msg.type === 'started') {
    setConnState('live');
    els.stopBtn.disabled = false;
    els.sessionLabel.textContent = `Session ${msg.session_id.slice(0, 8)}`;
    els.videoEmpty.style.display = 'none';
    els.videoFrame.style.display = 'block';
  } else if (msg.type === 'frame') {
    els.videoFrame.src = `data:image/jpeg;base64,${msg.image}`;
    updateTelemetry(msg.info);
  } else if (msg.type === 'ended') {
    setConnState('idle');
    els.stopBtn.disabled = true;
    els.startBtn.disabled = false;
    els.sessionLabel.textContent = 'Stream ended';
  } else if (msg.type === 'error') {
    setConnState('idle');
    els.startBtn.disabled = false;
    alert(`Stream error: ${msg.message}`);
  }
}

function setConnState(state) {
  els.connStatus.dataset.state = state;
  const label = { idle: 'OFFLINE', connecting: 'CONNECTING', live: 'LIVE', demo: 'DEMO' }[state] || 'OFFLINE';
  els.connStatus.querySelector('.status-text').textContent = label;
}

function resetUI() {
  els.alertTicker.innerHTML = '<div class="ticker-empty">No anomalies detected yet. System nominal.</div>';
  els.peopleCount.textContent = '0';
  els.fpsValue.textContent = '0.0';
  ['meterMag', 'meterStd', 'meterEnt'].forEach(k => els[k].style.width = '0%');
  ['valMag', 'valStd', 'valEnt'].forEach(k => els[k].textContent = '0.000');
}

// ---------------- Telemetry update ----------------
function updateTelemetry(info) {
  els.peopleCount.textContent = info.people_count;
  els.fpsValue.textContent = info.fps.toFixed(1);

  // meters are scaled heuristically for visual range; raw values shown alongside
  setMeter(els.meterMag, els.valMag, info.mean_magnitude, 6);
  setMeter(els.meterStd, els.valStd, info.motion_std, 6);
  setMeter(els.meterEnt, els.valEnt, info.direction_entropy, 1);

  if (info.calibrating) {
    els.calibOverlay.hidden = false;
    const pct = Math.min(100, (info.frame_idx / calibTotalFrames) * 100);
    els.calibFill.style.width = `${pct}%`;
    els.calibText.textContent = `Calibrating baseline… frame ${info.frame_idx}`;
  } else {
    els.calibOverlay.hidden = true;
  }

  if (info.anomalies && info.anomalies.length) {
    info.anomalies.forEach(a => pushAlert(a, info.frame_idx));
  }
}

function setMeter(meterEl, valueEl, value, maxScale) {
  const pct = Math.max(0, Math.min(100, (value / maxScale) * 100));
  meterEl.style.width = `${pct}%`;
  valueEl.textContent = value.toFixed(3);
}

function pushAlert(anomaly, frameIdx) {
  // remove empty-state placeholder if present
  const empty = els.alertTicker.querySelector('.ticker-empty');
  if (empty) empty.remove();

  const card = document.createElement('div');
  card.className = 'alert-card';
  card.dataset.type = anomaly.type;
  const time = new Date().toLocaleTimeString('en-GB');
  card.innerHTML = `
    <div class="alert-card-top">
      <span>${anomaly.label}</span>
      <span class="alert-card-score">${(anomaly.score * 100).toFixed(0)}%</span>
    </div>
    <div class="alert-card-meta">frame ${frameIdx} · ${time}${anomaly.detail ? ' · ' + anomaly.detail : ''}</div>
  `;
  els.alertTicker.prepend(card);

  // cap ticker length for performance
  const cards = els.alertTicker.querySelectorAll('.alert-card');
  if (cards.length > 40) cards[cards.length - 1].remove();
}
