// ===== THEME SYSTEM =====
function toggleTheme() {
  document.body.classList.toggle("light");

  if (document.body.classList.contains("light")) {
    localStorage.setItem("theme", "light");
  } else {
    localStorage.setItem("theme", "dark");
  }
}

// load saved theme
window.addEventListener("DOMContentLoaded", () => {
  const saved = localStorage.getItem("theme");
  if (saved === "light") {
    document.body.classList.add("light");
  }
});
const SID = 'sess_' + Math.random().toString(36).slice(2, 10);
let allImages = [];
let splitData = null;
let modelData = null;
let anomalyData = null;
let screeChart = null, cumChart = null, evalBarChart = null, kCurveChart = null, projChart = null, anomChart = null;

function switchTab(name) {
  const names = ['data','split','train','results','evaluate','reconstruct','anomaly','project'];
  document.querySelectorAll('.tab').forEach((el, i) => el.classList.toggle('active', names[i] === name));
  document.querySelectorAll('.panel').forEach(el => el.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
}

const dropZone = document.getElementById('dropZone');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('over'));
dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.classList.remove('over'); handleFiles(e.dataTransfer.files); });

async function handleFiles(files) {
  if (!files || files.length === 0) return;
  const validFiles = Array.from(files).filter(f => f.type.startsWith('image/') && f.size > 0);
  if (validFiles.length === 0) return;

  document.getElementById('uploadProgress').style.display = '';
  document.getElementById('uploadFill').style.width = '10%';
  document.getElementById('uploadLabel').textContent = `Uploading ${validFiles.length} file(s)...`;

  const fd = new FormData();
  fd.append('sid', SID);
  validFiles.forEach(f => fd.append('images', f));

  try {
    const res = await fetch('/upload', { method: 'POST', body: fd });
    const data = await res.json();
    document.getElementById('uploadFill').style.width = '100%';
    document.getElementById('uploadLabel').textContent = `Added ${data.added.length} image(s). Total: ${data.total}`;
    data.added.forEach(img => allImages.push(img));
    renderGrid();
    updateMetrics();
    previewSplit();
    setTimeout(() => { document.getElementById('uploadProgress').style.display = 'none'; }, 1500);
  } catch (e) {
    document.getElementById('uploadLabel').textContent = 'Upload failed: ' + e.message;
  }
}

function renderGrid() {
  const grid = document.getElementById('imgGrid');
  const card = document.getElementById('imgGridCard');
  card.style.display = allImages.length > 0 ? '' : 'none';
  grid.innerHTML = '';
  allImages.forEach(im => {
    const d = document.createElement('div');
    d.className = 'img-thumb';
    d.innerHTML = `<img src="data:image/png;base64,${im.thumb}" title="${im.name}" alt="${im.name}">
                   <button class="rm-btn" onclick="removeImage('${im.id}')">✕</button>`;
    grid.appendChild(d);
  });
  document.getElementById('confirmSplitBtn').disabled = allImages.length < 2;
}

async function removeImage(id) {
  await fetch('/remove', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({sid: SID, id}) });
  allImages = allImages.filter(x => x.id !== id);
  renderGrid();
  updateMetrics();
  previewSplit();
}

async function clearAll() {
  await fetch('/clear', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({sid: SID}) });
  allImages = [];
  splitData = null; modelData = null;
  renderGrid(); updateMetrics(); previewSplit();
  resetModelUI();
}

function updateSizeLabel() {
  const v = document.getElementById('sizeSlider').value;
  document.getElementById('sizeVal').textContent = v + '×' + v;
  updateMetrics();
}

function updateMetrics() {
  const n = allImages.length;
  const sz = parseInt(document.getElementById('sizeSlider').value);
  const isRGB = document.getElementById('colorMode').value === 'rgb';
  const dims = sz * sz * (isRGB ? 3 : 1);
  document.getElementById('m-imgs').textContent = n;
  document.getElementById('m-size').textContent = n > 0 ? sz + '×' + sz : '—';
  document.getElementById('m-dims').textContent = n > 0 ? dims : '—';
  document.getElementById('m-mat').textContent  = n > 0 ? n + '×' + dims : '—';
}

function previewSplit() {
  const n = allImages.length;
  const pct = parseInt(document.getElementById('splitSlider').value);
  document.getElementById('splitVal').textContent = pct;
  const nTrain = Math.max(1, Math.round(n * pct / 100));
  const nTest = n - nTrain;
  document.getElementById('trainCount').textContent = nTrain;
  document.getElementById('testCount').textContent = nTest;
  document.getElementById('confirmSplitBtn').disabled = n < 2;
  renderMiniGrid('trainGrid', allImages.slice(0, nTrain), 'is-train');
  renderMiniGrid('testGrid',  allImages.slice(nTrain), 'is-test');
}

function renderMiniGrid(id, imgs, cls) {
  const g = document.getElementById(id);
  if (imgs.length === 0) { g.innerHTML = '<p class="muted-text">None</p>'; return; }
  g.innerHTML = '';
  imgs.forEach(im => {
    const d = document.createElement('div');
    d.className = 'img-thumb ' + cls;
    d.innerHTML = `<img src="data:image/png;base64,${im.thumb}" alt="${im.name}">`;
    g.appendChild(d);
  });
}

async function confirmSplit() {
  const res = await fetch('/split', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      sid: SID,
      pct: parseInt(document.getElementById('splitSlider').value),
      strategy: document.getElementById('splitStrategy').value
    })
  });
  splitData = await res.json();
  document.getElementById('trainBtn').disabled = false;
  document.getElementById('trainOnLabel').textContent =
    `Training on ${splitData.train_idx.length} images, testing on ${splitData.test_idx.length} images`;
  document.getElementById('modelStatus').className = 'status-pill split';
  document.getElementById('modelStatus').textContent = '● Split ready';
  logMsg('Split confirmed: ' + splitData.train_idx.length + ' train, ' + splitData.test_idx.length + ' test', 'ok');
  switchTab('train');
}

async function runTraining() {
  if (!splitData) return;
  document.getElementById('trainLog').innerHTML = '';
  logMsg('Starting PCA training...', 'info');
  document.getElementById('trainFill').style.width = '5%';
  document.getElementById('trainBtn').disabled = true;

  const sz = parseInt(document.getElementById('sizeSlider').value);
  const isRGB = document.getElementById('colorMode').value === 'rgb';
  const k = parseInt(document.getElementById('nCompSlider').value);
  const center = document.getElementById('centerMode').value === 'yes';

  logMsg(`Params: size=${sz}×${sz}  mode=${isRGB?'RGB':'Gray'}  k=${k}  centering=${center}`);
  document.getElementById('trainFill').style.width = '20%';

  try {
    const res = await fetch('/train', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ sid: SID, sz, is_rgb: isRGB, k, center })
    });
    const data = await res.json();
    if (data.error) { logMsg('Error: ' + data.error, 'err'); document.getElementById('trainBtn').disabled = false; return; }

    document.getElementById('trainFill').style.width = '100%';
    modelData = data;
    logMsg(`Done! ${data.k || 0} components explain ${data.final_var || 0}% variance`, 'ok');

logMsg(
  `Top PC explains ${data.top_var || 'N/A'}% · Matrix: ${data.N || 'N/A'}×${data.D || 'N/A'}`,
  'ok'
);

    document.getElementById('modelStatus').className = 'status-pill trained';
    document.getElementById('modelStatus').textContent = '● Trained';
    document.getElementById('saveBtn').disabled = false;
    document.getElementById('evalBtn').disabled = false;

    populateResults(data);
    await loadEigenfaces();
    populateReconSelect();
    await runReconstruction();
    await loadAnomalyScores();
    populatePCSelectors(data.k);

    switchTab('results');
  } catch(e) {
    logMsg('Training error: ' + e.message, 'err');
  }
  document.getElementById('trainBtn').disabled = false;
}

function populateResults(data) {
  document.getElementById('r-comp').textContent  = data.k || '—';
  document.getElementById('r-var').textContent   = data.final_var ? data.final_var + '%' : '—';

  // ✅ SAFE eigenvalue access
  if (data.eigenvals && data.eigenvals.length > 0) {
    document.getElementById('r-eig').textContent = data.eigenvals[0].toFixed(3);
  } else {
    document.getElementById('r-eig').textContent = '—';
  }

  // ✅ SAFE compression %
  if (data.D && data.k) {
    document.getElementById('r-comp2').textContent = ((data.k / data.D) * 100).toFixed(1) + '%';
  } else {
    document.getElementById('r-comp2').textContent = '—';
  }

  // ✅ SAFE arrays
  const varRatios = data.var_ratios || [];
  const cumVar = data.cum_var || [];

  const labels = varRatios.map((_, i) => 'PC' + (i+1));
  const screeData = varRatios.map(v => parseFloat((v*100).toFixed(2)));
  const cumData   = cumVar.map(v => parseFloat((v*100).toFixed(2)));

  if (screeChart) screeChart.destroy();
  screeChart = new Chart(document.getElementById('screeChart'), {
    type: 'bar',
    data: { labels, datasets: [{ label: 'Variance %', data: screeData, backgroundColor: '#3b82f6', borderRadius: 4 }] },
    options: { responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { ticks: { color: '#8b93a8' }, grid: { color: 'rgba(255,255,255,0.04)' } },
                y: { ticks: { color: '#8b93a8' }, grid: { color: 'rgba(255,255,255,0.04)' } } }
    }
  });

  if (cumChart) cumChart.destroy();
  cumChart = new Chart(document.getElementById('cumChart'), {
    type: 'line',
    data: { labels, datasets: [{ label: 'Cumulative %', data: cumData, borderColor: '#4ade80', fill: true }] },
    options: { responsive: true, maintainAspectRatio: false }
  });
}

async function loadEigenfaces() {
  const res = await fetch('/eigenfaces', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({sid: SID}) });
  const data = await res.json();
  const row = document.getElementById('eigenRow');
  row.innerHTML = '';
  data.eigenfaces.forEach(ef => {
    const div = document.createElement('div');
    div.className = 'eigen-item';
    div.innerHTML = `<img src="data:image/png;base64,${ef.image}" alt="PC${ef.pc}"><span>PC${ef.pc}</span>`;
    row.appendChild(div);
  });
}

function populateReconSelect() {
  const sel = document.getElementById('reconSelect');
  sel.innerHTML = allImages.map((im, i) =>
    `<option value="${i}">${im.name}${splitData && splitData.train_idx.includes(i) ? ' [train]' : ' [test]'}</option>`
  ).join('');
  const slider = document.getElementById('reconSlider');
  slider.max = modelData.k;
  slider.value = Math.min(modelData.k, 10);
  document.getElementById('reconVal').textContent = slider.value;
}

async function runReconstruction() {
  if (!modelData) return;
  const idx = parseInt(document.getElementById('reconSelect').value) || 0;
  const nComp = parseInt(document.getElementById('reconSlider').value);
  document.getElementById('reconVal').textContent = nComp;
  const res = await fetch('/reconstruct', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ sid: SID, img_idx: idx, n_comp: nComp })
  });
  const data = await res.json();
  if (data.error) return;
  document.getElementById('origImg').src  = 'data:image/png;base64,' + data.orig_b64;
  document.getElementById('reconImg').src = 'data:image/png;base64,' + data.recon_b64;
  document.getElementById('residImg').src = 'data:image/png;base64,' + data.resid_b64;
  document.getElementById('origDims').textContent  = data.dims;
  document.getElementById('reconUsed').textContent = nComp;
  document.getElementById('reconMSE').textContent  = data.mse.toExponential(4);
  document.getElementById('reconOutput').style.display = '';
}

async function runEvaluation() {
  if (!modelData) return;
  document.getElementById('evalBtn').disabled = true;
  document.getElementById('evalBtn').textContent = 'Evaluating...';
  const res = await fetch('/evaluate', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({sid: SID}) });
  const d = await res.json();
  if (d.error) { document.getElementById('evalBtn').disabled = false; document.getElementById('evalBtn').textContent = 'Run Evaluation'; return; }

  document.getElementById('evalMetrics').style.display = 'grid';
  document.getElementById('e-trainMSE').textContent = d.avg_train.toExponential(3);
  document.getElementById('e-testMSE').textContent  = d.avg_test !== null ? d.avg_test.toExponential(3) : 'N/A';
  document.getElementById('e-gap').textContent = d.gap !== null ? d.gap + '%' : 'N/A';
  const vEl = document.getElementById('e-verdict');
  vEl.textContent = d.verdict;
  vEl.className = 'stat-val small ' + d.verdict_class;

  document.getElementById('evalCharts').style.display = 'grid';
  if (evalBarChart) evalBarChart.destroy();
  evalBarChart = new Chart(document.getElementById('evalBarChart'), {
    type: 'bar',
    data: { labels: d.bar_labels, datasets: [{ label: 'MSE', data: d.bar_data, backgroundColor: d.bar_colors, borderRadius: 3 }] },
    options: { responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { ticks: { color: '#8b93a8', font: { size: 10 }, maxRotation: 45 }, grid: { color: 'rgba(255,255,255,0.04)' } },
                y: { ticks: { color: '#8b93a8' }, grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: 'MSE', color: '#8b93a8' } } } }
  });

  const kDatasets = [{ label: 'Train MSE', data: d.k_train, borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.07)', fill: true, tension: 0.3, pointRadius: 4 }];
  if (d.k_test.length > 0) kDatasets.push({ label: 'Test MSE', data: d.k_test, borderColor: '#4ade80', backgroundColor: 'rgba(74,222,128,0.07)', fill: true, tension: 0.3, pointRadius: 4 });
  if (kCurveChart) kCurveChart.destroy();
  kCurveChart = new Chart(document.getElementById('kCurveChart'), {
    type: 'line',
    data: { labels: d.k_values, datasets: kDatasets },
    options: { responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: true, labels: { color: '#8b93a8' } } },
      scales: { x: { ticks: { color: '#8b93a8' }, grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: 'k (components)', color: '#8b93a8' } },
                y: { ticks: { color: '#8b93a8' }, grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: 'MSE', color: '#8b93a8' } } } }
  });

  document.getElementById('evalInsightCard').style.display = '';
  let ins = `<strong>Train MSE = ${d.avg_train.toExponential(2)}</strong> — How well the model reconstructs its training images. Lower is better.<br>`;
  if (d.avg_test !== null) {
    ins += `<strong>Test MSE = ${d.avg_test.toExponential(2)}</strong> — How well it works on unseen images. Gap = ${d.gap}%.<br>`;
    if (d.gap < 15) ins += `<span class="verdict-good">Small gap — model generalises well.</span><br>`;
    else if (d.gap < 40) ins += `<span class="verdict-warn">Moderate gap — try more images or a larger k.</span><br>`;
    else ins += `<span class="verdict-bad">Large gap — model is overfitting. Try more training images.</span><br>`;
  }
  ins += `<strong>k-curve:</strong> As k increases, error decreases. The point the test curve flattens is your optimal k. Current k = ${d.k}.`;
  document.getElementById('evalInsight').innerHTML = ins;

  document.getElementById('evalBtn').disabled = false;
  document.getElementById('evalBtn').textContent = 'Run Evaluation';
}

async function loadAnomalyScores() {
  const res = await fetch('/anomaly_scores', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({sid: SID}) });
  anomalyData = await res.json();
  document.getElementById('threshSlider').disabled = false;
  drawAnomalyChart(2.0);
  document.getElementById('anomalyScoreCard').style.display = '';
}

function updateThreshold() {
  const mult = parseInt(document.getElementById('threshSlider').value) / 10;
  document.getElementById('threshVal').textContent = mult.toFixed(1);
  drawAnomalyChart(mult);
}

function drawAnomalyChart(mult) {
  if (!anomalyData) return;
  const thresh = anomalyData.mean_mse * mult;
  const colors = anomalyData.scores.map(s => s > thresh ? '#f87171' : '#3b82f6');
  const threshLine = new Array(anomalyData.scores.length).fill(thresh);

  if (anomChart) anomChart.destroy();
  anomChart = new Chart(document.getElementById('anomalyChart'), {
    type: 'bar',
    data: {
      labels: anomalyData.labels,
      datasets: [
        { label: 'Reconstruction Error', data: anomalyData.scores, backgroundColor: colors, borderRadius: 3 },
        { label: 'Threshold', data: threshLine, type: 'line', borderColor: '#fbbf24', borderDash: [6,3], borderWidth: 2, pointRadius: 0, fill: false }
      ]
    },
    options: { responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: true, labels: { color: '#8b93a8' } }, tooltip: { callbacks: { label: ctx => ctx.datasetIndex === 1 ? `Threshold: ${thresh.toExponential(3)}` : `MSE: ${anomalyData.scores[ctx.dataIndex].toExponential(3)}${anomalyData.scores[ctx.dataIndex] > thresh ? ' ⚠ ANOMALY' : ' ✓ Normal'}` } } },
      scales: { x: { ticks: { color: '#8b93a8', font: { size: 10 }, maxRotation: 45 }, grid: { color: 'rgba(255,255,255,0.04)' } },
                y: { ticks: { color: '#8b93a8' }, grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: 'MSE', color: '#8b93a8' } } } }
  });
}

async function testAnomaly(input) {
  if (!modelData || !input.files[0]) return;
  const mult = parseInt(document.getElementById('threshSlider').value) / 10;
  const fd = new FormData();
  fd.append('sid', SID);
  fd.append('image', input.files[0]);
  const res = await fetch('/test_anomaly', { method: 'POST', body: fd });
  const d = await res.json();
  if (d.error) { alert(d.error); return; }
  const thresh = d.mean_mse * mult;
  const isAnom = d.mse > thresh;
  document.getElementById('aOrigImg').src  = 'data:image/png;base64,' + d.orig_b64;
  document.getElementById('aReconImg').src = 'data:image/png;base64,' + d.recon_b64;
  document.getElementById('aResidImg').src = 'data:image/png;base64,' + d.resid_b64;
  document.getElementById('anomalyVerdict').innerHTML =
    `<strong>File:</strong> ${d.filename}<br>
     <strong>MSE:</strong> ${d.mse.toExponential(4)}<br>
     <strong>Threshold:</strong> ${thresh.toExponential(4)}<br>
     <strong>Ratio:</strong> ${d.ratio}× mean train error<br>
     <strong>Result:</strong> <span class="${isAnom ? 'verdict-bad' : 'verdict-good'}" style="font-size:15px">
       ${isAnom ? '⚠ ANOMALY — this image looks unusual' : '✓ NORMAL — this image looks similar to training data'}
     </span>`;
  document.getElementById('anomalyResultCard').style.display = '';
  input.value = '';
}

function populatePCSelectors(k) {
  const opts = Array.from({length: k}, (_, i) => `<option value="${i}">PC ${i+1}</option>`).join('');
  document.getElementById('pcX').innerHTML = opts;
  document.getElementById('pcY').innerHTML = opts;
  document.getElementById('pcY').value = Math.min(1, k - 1);
  runProjection();
}

async function runProjection() {
  if (!modelData) return;
  const xi = parseInt(document.getElementById('pcX').value);
  const yi = parseInt(document.getElementById('pcY').value);
  const res = await fetch('/projection', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({sid: SID, xi, yi}) });
  const d = await res.json();
  if (d.error) return;
  const trainPts = d.points.filter(p => p.is_train).map(p => ({ x: p.x, y: p.y, label: p.name }));
  const testPts  = d.points.filter(p => !p.is_train).map(p => ({ x: p.x, y: p.y, label: p.name }));
  if (projChart) projChart.destroy();
  projChart = new Chart(document.getElementById('projChart'), {
    type: 'scatter',
    data: {
      datasets: [
        { label: 'Training', data: trainPts, backgroundColor: '#3b82f6', pointRadius: 8, pointHoverRadius: 10 },
        { label: 'Test',     data: testPts,  backgroundColor: '#4ade80', pointRadius: 8, pointHoverRadius: 10 }
      ]
    },
    options: { responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => `${ctx.raw.label}  (${ctx.raw.x.toFixed(3)}, ${ctx.raw.y.toFixed(3)})` } } },
      scales: { x: { ticks: { color: '#8b93a8' }, grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: 'PC ' + (xi+1), color: '#8b93a8' } },
                y: { ticks: { color: '#8b93a8' }, grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: 'PC ' + (yi+1), color: '#8b93a8' } } } }
  });
}

async function saveModel() {
  const res = await fetch('/save_model', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({sid: SID}) });
  const data = await res.json();
  const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'pca-model.json'; a.click();
  URL.revokeObjectURL(url);
}

async function loadModel(input) {
  const file = input.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append('sid', SID);
  fd.append('model_file', file);
  const res = await fetch('/load_model', { method: 'POST', body: fd });
  const data = await res.json();
  if (data.error) { alert('Error: ' + data.error); return; }
  modelData = data;
  document.getElementById('modelStatus').className = 'status-pill trained';
  document.getElementById('modelStatus').textContent = '● Loaded';
  document.getElementById('saveBtn').disabled = false;
  document.getElementById('evalBtn').disabled = false;
  populateResults(data);
  await loadEigenfaces();
  alert('Model loaded. Upload matching images and go to Reconstruct tab.');
  input.value = '';
}

function logMsg(msg, cls = '') {
  const el = document.getElementById('trainLog');
  const div = document.createElement('div');
  if (cls) div.className = 'log-' + cls;
  div.textContent = msg;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

function resetModelUI() {
  modelData = null; anomalyData = null; splitData = null;
  [screeChart, cumChart, evalBarChart, kCurveChart, projChart, anomChart].forEach(c => { if (c) c.destroy(); });
  screeChart = cumChart = evalBarChart = kCurveChart = projChart = anomChart = null;
  document.getElementById('trainLog').innerHTML = '<span class="log-info">Confirm split first...</span>';
  document.getElementById('trainFill').style.width = '0%';
  document.getElementById('trainBtn').disabled = true;
  document.getElementById('confirmSplitBtn').disabled = true;
  document.getElementById('evalBtn').disabled = true;
  document.getElementById('saveBtn').disabled = true;
  document.getElementById('eigenRow').innerHTML = '<span class="muted-text">Train to see eigenfaces</span>';
  document.getElementById('reconOutput').style.display = 'none';
  document.getElementById('evalMetrics').style.display = 'none';
  document.getElementById('evalCharts').style.display = 'none';
  document.getElementById('evalInsightCard').style.display = 'none';
  document.getElementById('anomalyScoreCard').style.display = 'none';
  document.getElementById('anomalyResultCard').style.display = 'none';
  document.getElementById('threshSlider').disabled = true;
  document.getElementById('modelStatus').className = 'status-pill untrained';
  document.getElementById('modelStatus').textContent = '● Untrained';
  ['r-comp','r-var','r-eig','r-comp2','e-trainMSE','e-testMSE','e-gap','e-verdict'].forEach(id => {
    const el = document.getElementById(id); if (el) el.textContent = '—';
  });
}

function resetAll() {
  clearAll();
  switchTab('data');
}
