(() => {
  const UI = window.NeuralUI;
  const palette = ['#ff6b6b', '#4dabf7', '#ffd93d', '#6ddccf', '#b388eb', '#ff9770', '#84dcc6', '#95a9f9'];

  const boardCanvas = document.getElementById('matchBoard');
  const boardCtx = boardCanvas.getContext('2d');
  const errorCanvas = document.getElementById('errorCanvas');
  const errorCtx = errorCanvas.getContext('2d');
  const scoreCanvas = document.getElementById('scoreCanvas');
  const scoreCtx = scoreCanvas.getContext('2d');
  const diversityCanvas = document.getElementById('diversityCanvas');
  const diversityCtx = diversityCanvas.getContext('2d');

  const patternNameEl = document.getElementById('patternName');
  const patternTransformsEl = document.getElementById('patternTransforms');
  const patternNoiseEl = document.getElementById('patternNoise');
  const bestSwapEl = document.getElementById('bestSwap');
  const targetScoreEl = document.getElementById('targetScore');
  const modelScoreEl = document.getElementById('modelScore');
  const batchErrorEl = document.getElementById('batchError');
  const avgErrorEl = document.getElementById('avgError');
  const confidenceEl = document.getElementById('confidence');

  const totalBoardsEl = document.getElementById('totalBoards');
  const teacherAvgEl = document.getElementById('teacherAvg');
  const modelAvgEl = document.getElementById('modelAvg');
  const maeEl = document.getElementById('mae');
  const transformCountEl = document.getElementById('transformCount');
  const weightUpdatesEl = document.getElementById('weightUpdates');
  const patternListEl = document.getElementById('patternList');

  const startBtn = document.getElementById('startBtn');
  const pauseBtn = document.getElementById('pauseBtn');
  const resetBtn = document.getElementById('resetBtn');

  const lrEl = document.getElementById('learningRate');
  const lrValEl = document.getElementById('learningRateVal');
  const hiddenEl = document.getElementById('hiddenUnits');
  const hiddenValEl = document.getElementById('hiddenUnitsVal');
  const batchEl = document.getElementById('batchSize');
  const batchValEl = document.getElementById('batchSizeVal');
  const noiseEl = document.getElementById('noise');
  const noiseValEl = document.getElementById('noiseVal');
  const delayEl = document.getElementById('delayMs');
  const delayValEl = document.getElementById('delayMsVal');
  const burstEl = document.getElementById('burstFactor');
  const burstValEl = document.getElementById('burstFactorVal');

  let running = false;
  let boardState = { size: 6, cells: new Array(36).fill(0) };
  let boardHighlight = null;
  let patternOrder = [];
  let diversityHistory = [];
  const maxHistory = 240;
  const maxDiversity = 96;
  let errorHistory = [];
  let avgErrorHistory = [];
  let scoreHistoryTeacher = [];
  let scoreHistoryModel = [];

  function drawBoard() {
    const size = boardState.size;
    const cells = boardState.cells || [];
    const w = boardCanvas.width;
    const h = boardCanvas.height;
    const padding = 18;
    const cellSize = (Math.min(w, h) - padding * 2) / size;
    boardCtx.fillStyle = '#0e1117';
    boardCtx.fillRect(0, 0, w, h);

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const idx = y * size + x;
        const colorIdx = cells[idx] ?? 0;
        const color = palette[colorIdx % palette.length];
        const px = padding + x * cellSize;
        const py = padding + y * cellSize;
        boardCtx.fillStyle = color;
        boardCtx.fillRect(px, py, cellSize - 1, cellSize - 1);
      }
    }

    if (boardHighlight) {
      boardCtx.lineWidth = 3;
      boardCtx.strokeStyle = '#ffffff';
      boardHighlight.forEach(idx => {
        const x = idx % size;
        const y = (idx / size) | 0;
        const px = padding + x * cellSize + 1.5;
        const py = padding + y * cellSize + 1.5;
        boardCtx.strokeRect(px, py, cellSize - 3, cellSize - 3);
      });
    }
  }

  function drawLineChart(ctx, series, options = {}) {
    const { color = '#ff8133', width = 2, min = 0, max = 1, clear = true } = options;
    const w = ctx.canvas.width;
    const h = ctx.canvas.height;
    if (clear) ctx.clearRect(0, 0, w, h);
    if (!series.length) return;
    const n = series.length;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    series.forEach((v, i) => {
      const x = (i / Math.max(1, n - 1)) * w;
      const y = h - ((v - min) / Math.max(1e-6, (max - min))) * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  function drawScoreChart() {
    if (!scoreHistoryTeacher.length) {
      scoreCtx.clearRect(0, 0, scoreCanvas.width, scoreCanvas.height);
      return;
    }
    const maxVal = Math.max(1, ...scoreHistoryTeacher, ...scoreHistoryModel);
    drawLineChart(scoreCtx, scoreHistoryTeacher, { color: '#ffa94d', min: 0, max: maxVal });
    drawLineChart(scoreCtx, scoreHistoryModel, { color: '#74c0fc', min: 0, max: maxVal, width: 2, clear: false });
  }

  function drawErrorChart() {
    if (!errorHistory.length) {
      errorCtx.clearRect(0, 0, errorCanvas.width, errorCanvas.height);
      return;
    }
    const maxVal = Math.max(1e-3, ...errorHistory, ...avgErrorHistory);
    drawLineChart(errorCtx, errorHistory, { color: '#ff922b', min: 0, max: maxVal });
    drawLineChart(errorCtx, avgErrorHistory, { color: '#339af0', min: 0, max: maxVal, width: 2, clear: false });
  }

  function drawDiversityHeatmap() {
    const w = diversityCanvas.width;
    const h = diversityCanvas.height;
    diversityCtx.clearRect(0, 0, w, h);
    if (!patternOrder.length || !diversityHistory.length) return;
    const cols = diversityHistory.length;
    const rows = patternOrder.length;
    const cellW = w / cols;
    const cellH = h / rows;
    diversityHistory.forEach((frame, col) => {
      frame.forEach((value, row) => {
        const intensity = Math.min(1, value);
        const hue = (row / Math.max(1, rows)) * 360;
        const alpha = 0.15 + intensity * 0.85;
        diversityCtx.fillStyle = `hsla(${hue}, 74%, 54%, ${alpha})`;
        diversityCtx.fillRect(col * cellW, row * cellH, cellW + 1, cellH + 1);
      });
    });
    diversityCtx.strokeStyle = '#111';
    diversityCtx.lineWidth = 1;
    diversityCtx.strokeRect(0, 0, w, h);
  }

  function updatePatternList(counts) {
    patternListEl.innerHTML = '';
    const total = Object.values(counts).reduce((sum, v) => sum + v, 0) || 1;
    patternOrder.forEach(name => {
      const li = document.createElement('li');
      const count = counts[name] || 0;
      const pct = ((count / total) * 100).toFixed(1);
      const label = document.createElement('span');
      label.textContent = name;
      const value = document.createElement('span');
      value.textContent = `${count} · ${pct}%`;
      li.appendChild(label);
      li.appendChild(value);
      patternListEl.appendChild(li);
    });
  }

  function describeSwap(move, size) {
    if (!move) return '—';
    const { from, to } = move;
    if (from == null || to == null) return '—';
    const fx = (from % size) + 1;
    const fy = Math.floor(from / size) + 1;
    const tx = (to % size) + 1;
    const ty = Math.floor(to / size) + 1;
    return `${fx},${fy} ↔ ${tx},${ty}`;
  }

  function setRunning(state) {
    running = state;
    startBtn.disabled = running;
    pauseBtn.disabled = !running;
  }

  const worker = new Worker('worker.js');

  worker.onmessage = ev => {
    const msg = ev.data;
    if (!msg) return;
    if (msg.type === 'status') {
      setRunning(!!msg.running);
    } else if (msg.type === 'snapshot') {
      boardState = msg.board;
      boardHighlight = msg.highlight || null;
      drawBoard();

      const patternLabel = msg.pattern?.variant
        ? `${msg.pattern.name} — ${msg.pattern.variant}`
        : msg.pattern?.name || '—';
      patternNameEl.textContent = patternLabel;
      patternTransformsEl.textContent = (msg.pattern?.transforms && msg.pattern.transforms.length)
        ? msg.pattern.transforms.join(' · ')
        : 'none';
      patternNoiseEl.textContent = `${Math.round((msg.pattern?.noise ?? 0) * 100)}%`;

      bestSwapEl.textContent = describeSwap(msg.bestMove, boardState.size);
      targetScoreEl.textContent = (msg.targetScore ?? 0).toFixed(2);
      modelScoreEl.textContent = (msg.modelScore ?? 0).toFixed(2);
      batchErrorEl.textContent = (msg.batchError ?? 0).toFixed(4);
      avgErrorEl.textContent = (msg.runningError ?? 0).toFixed(4);
      confidenceEl.textContent = (msg.confidence ?? 0).toFixed(3);

      totalBoardsEl.textContent = msg.stats?.boards ?? 0;
      teacherAvgEl.textContent = (msg.stats?.teacherAvg ?? 0).toFixed(2);
      modelAvgEl.textContent = (msg.stats?.modelAvg ?? 0).toFixed(2);
      maeEl.textContent = (msg.stats?.mae ?? 0).toFixed(4);
      transformCountEl.textContent = msg.stats?.transformCount ?? 0;
      weightUpdatesEl.textContent = msg.stats?.weightUpdates ?? 0;

      if (Array.isArray(msg.patternOrder)) {
        patternOrder = msg.patternOrder;
      }
      if (msg.patternCounts) {
        updatePatternList(msg.patternCounts);
      }
      if (Array.isArray(msg.errorHistory)) {
        errorHistory = msg.errorHistory.slice(-maxHistory);
      }
      if (Array.isArray(msg.avgErrorHistory)) {
        avgErrorHistory = msg.avgErrorHistory.slice(-maxHistory);
      }
      if (Array.isArray(msg.scoreHistoryTeacher)) {
        scoreHistoryTeacher = msg.scoreHistoryTeacher.slice(-maxHistory);
      }
      if (Array.isArray(msg.scoreHistoryModel)) {
        scoreHistoryModel = msg.scoreHistoryModel.slice(-maxHistory);
      }
      if (Array.isArray(msg.diversityFrame)) {
        diversityHistory.push(msg.diversityFrame);
        if (diversityHistory.length > maxDiversity) diversityHistory.shift();
      }
      drawErrorChart();
      drawScoreChart();
      drawDiversityHeatmap();
    } else if (msg.type === 'reset') {
      errorHistory = [];
      avgErrorHistory = [];
      scoreHistoryTeacher = [];
      scoreHistoryModel = [];
      diversityHistory = [];
      patternOrder = msg.patternOrder || patternOrder;
      updatePatternList(msg.patternCounts || {});
      drawBoard();
      drawErrorChart();
      drawScoreChart();
      drawDiversityHeatmap();
    }
  };

  startBtn.addEventListener('click', () => worker.postMessage({ type: 'start' }));
  pauseBtn.addEventListener('click', () => worker.postMessage({ type: 'pause' }));
  resetBtn.addEventListener('click', () => worker.postMessage({ type: 'reset' }));

  if (UI) {
    UI.initTabs();
    const configHandler = key => value => worker.postMessage({ type: 'configure', config: { [key]: value } });
    UI.bindRangeControl(lrEl, { valueEl: lrValEl, format: v => (+v).toFixed(2), onCommit: configHandler('learningRate') });
    UI.bindRangeControl(hiddenEl, { valueEl: hiddenValEl, format: v => Math.round(+v), onCommit: configHandler('hiddenUnits') });
    UI.bindRangeControl(batchEl, { valueEl: batchValEl, format: v => Math.round(+v), onCommit: configHandler('batchSize') });
    UI.bindRangeControl(noiseEl, { valueEl: noiseValEl, format: v => Math.round(+v), onCommit: v => configHandler('noise')(v / 100) });
    UI.bindRangeControl(delayEl, { valueEl: delayValEl, format: v => Math.round(+v), onCommit: configHandler('delayMs') });
    UI.bindRangeControl(burstEl, { valueEl: burstValEl, format: v => Math.round(+v), onCommit: configHandler('burstFactor') });
  }

  drawBoard();
  drawErrorChart();
  drawScoreChart();
  drawDiversityHeatmap();
})();
