// Created: 2026-04-21 00:00:00

const API_URL = "http://127.0.0.1:8000";

const canvas     = document.getElementById("canvas");
const ctx        = canvas.getContext("2d");
const digitEl    = document.getElementById("digit");
const confEl     = document.getElementById("confidence");
const statusEl   = document.getElementById("status");
const barsEl     = document.getElementById("bars-container");
const btnClear   = document.getElementById("btn-clear");
const btnPredict = document.getElementById("btn-predict");

// ── Build probability bar rows ──────────────────────────────────────────────
const barFills = [];
const barPcts  = [];
const barDigitLabels = [];

for (let d = 0; d < 10; d++) {
  const row   = document.createElement("div");
  row.className = "bar-row";

  const lbl   = document.createElement("div");
  lbl.className = "bar-digit";
  lbl.textContent = d;

  const track = document.createElement("div");
  track.className = "bar-track";

  const fill  = document.createElement("div");
  fill.className = "bar-fill";
  track.appendChild(fill);

  const pct   = document.createElement("div");
  pct.className = "bar-pct";
  pct.textContent = "0%";

  row.append(lbl, track, pct);
  barsEl.appendChild(row);

  barFills.push(fill);
  barPcts.push(pct);
  barDigitLabels.push(lbl);
}

// ── Canvas drawing ──────────────────────────────────────────────────────────
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = "black";
ctx.lineWidth = 20;
ctx.lineCap   = "round";
ctx.lineJoin  = "round";

let drawing = false;
let lastX = 0, lastY = 0;

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const src  = e.touches ? e.touches[0] : e;
  return [src.clientX - rect.left, src.clientY - rect.top];
}

function startDraw(e) {
  e.preventDefault();
  drawing = true;
  [lastX, lastY] = getPos(e);
  ctx.beginPath();
  ctx.arc(lastX, lastY, ctx.lineWidth / 2, 0, Math.PI * 2);
  ctx.fillStyle = "black";
  ctx.fill();
}

function draw(e) {
  e.preventDefault();
  if (!drawing) return;
  const [x, y] = getPos(e);
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();
  [lastX, lastY] = [x, y];
}

function endDraw(e) {
  e.preventDefault();
  if (!drawing) return;
  drawing = false;
  sendPredict();
}

canvas.addEventListener("mousedown",  startDraw);
canvas.addEventListener("mousemove",  draw);
canvas.addEventListener("mouseup",    endDraw);
canvas.addEventListener("mouseleave", endDraw);
canvas.addEventListener("touchstart", startDraw, { passive: false });
canvas.addEventListener("touchmove",  draw,      { passive: false });
canvas.addEventListener("touchend",   endDraw,   { passive: false });

// ── Clear ───────────────────────────────────────────────────────────────────
function clearCanvas() {
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  digitEl.textContent = "—";
  confEl.textContent  = "숫자를 그리고 예측을 누르세요";
  barFills.forEach((f, i) => {
    f.style.width = "0%";
    f.classList.remove("active");
    barPcts[i].textContent = "0%";
    barDigitLabels[i].classList.remove("active");
  });
}

btnClear.addEventListener("click", clearCanvas);

// ── Predict ─────────────────────────────────────────────────────────────────
async function sendPredict() {
  const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/png"));
  const form = new FormData();
  form.append("file", blob, "canvas.png");

  try {
    const res  = await fetch(`${API_URL}/predict`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`서버 오류: ${res.status}`);
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    setStatus(`오류: ${err.message}`);
  }
}

btnPredict.addEventListener("click", sendPredict);

function renderResult({ digit, confidence, probabilities }) {
  digitEl.textContent = digit;
  confEl.textContent  = `신뢰도: ${confidence.toFixed(1)}%`;

  probabilities.forEach((pct, i) => {
    barFills[i].style.width = `${pct}%`;
    barFills[i].classList.toggle("active", i === digit);
    barPcts[i].textContent = `${pct.toFixed(1)}%`;
    barDigitLabels[i].classList.toggle("active", i === digit);
  });
}

// ── Health check ─────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_URL}/health`);
    const { model_loaded } = await res.json();
    setStatus(model_loaded ? "모델 준비 완료 — 숫자를 그려보세요!" : "모델 로딩 중…");
  } catch {
    setStatus("서버에 연결할 수 없습니다. 백엔드를 실행해주세요.");
  }
}

function setStatus(msg) {
  statusEl.textContent = msg;
}

checkHealth();
