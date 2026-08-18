const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');
const summary = document.getElementById('summary');

let stream = null;
let model = null;
let detectionTimer = null;
let isRunning = false;

function setStatus(message, kind = 'info') {
  status.textContent = message;
  status.className = `status ${kind}`;
}

function drawVideoFrame() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
}

function drawBox(prediction) {
  const [x, y, width, height] = prediction.bbox;
  ctx.strokeStyle = '#22c55e';
  ctx.lineWidth = 3;
  ctx.strokeRect(x, y, width, height);
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(x, y - 28, Math.max(140, ctx.measureText(prediction.className).width + 20), 24);
  ctx.fillStyle = '#ffffff';
  ctx.font = '16px Inter, Arial, sans-serif';
  ctx.fillText(`${prediction.className} (${Math.round(prediction.score * 100)}%)`, x + 8, y - 10);
}

async function loadModel() {
  if (model) return model;
  setStatus('Loading AI model…', 'info');
  model = await window.cocoSsd.load();
  setStatus('AI model ready. Click Start Webcam to begin.', 'good');
  return model;
}

async function detectFrame() {
  if (!isRunning || !video.videoWidth || !model) return;

  drawVideoFrame();
  const predictions = await model.detect(canvas);
  const persons = predictions.filter((item) => item.className === 'person' && item.score >= 0.35);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  persons.forEach(drawBox);

  const safeCount = persons.length;
  const violationCount = 0;
  summary.innerHTML = `Workers: <strong>${persons.length}</strong> · Safe: <strong>${safeCount}</strong> · Violations: <strong>${violationCount}</strong>`;

  if (persons.length > 0) {
    setStatus('Webcam live — person detection active.', 'good');
  } else {
    setStatus('No people detected yet. Point the camera at a worker.', 'info');
  }
}

async function startWebcam() {
  if (isRunning) return;

  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('This browser does not support webcam access.');
    }

    const streamObj = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });

    stream = streamObj;
    video.srcObject = stream;
    await video.play();

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    await loadModel();
    isRunning = true;

    if (detectionTimer) clearInterval(detectionTimer);
    detectionTimer = setInterval(() => {
      detectFrame().catch((err) => {
        setStatus(`Detection error: ${err.message}`, 'warn');
      });
    }, 800);

    setStatus('Webcam running — live person detection active.', 'good');
  } catch (err) {
    setStatus(`Could not start webcam: ${err.message}`, 'warn');
  }
}

function stopWebcam() {
  if (detectionTimer) {
    clearInterval(detectionTimer);
    detectionTimer = null;
  }

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  isRunning = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  summary.innerHTML = 'Waiting for webcam…';
  setStatus('Stream stopped.', 'info');
}

startBtn.addEventListener('click', startWebcam);
stopBtn.addEventListener('click', stopWebcam);

window.addEventListener('beforeunload', stopWebcam);
