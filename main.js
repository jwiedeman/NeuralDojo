// Neural Go self-play UI orchestration.
(() => {
  const { Board, BLACK, WHITE, xyOf } = window.GoEngine;

  const size = 9;
  const komi = 6.5;
  const board = new Board(size, komi);

  const cvs = document.getElementById('board');
  const ctx = cvs.getContext('2d');
  let lastMove = -1;

  const turnLabel = document.getElementById('turnLabel');
  const moveCountEl = document.getElementById('moveCount');
  const capBEl = document.getElementById('capB');
  const capWEl = document.getElementById('capW');
  const statusText = document.getElementById('statusText');
  const confNowEl = document.getElementById('confNow');

  const gameCountEl = document.getElementById('gameCount');
  const lastWinnerEl = document.getElementById('lastWinner');
  const lastScoreEl = document.getElementById('lastScore');

  const totalGamesEl = document.getElementById('totalGames');
  const blackWinsEl = document.getElementById('blackWins');
  const whiteWinsEl = document.getElementById('whiteWins');
  const blackWinRateEl = document.getElementById('blackWinRate');
  const predAccuracyEl = document.getElementById('predAccuracy');
  const avgConfidenceEl = document.getElementById('avgConfidence');
  const trainingStepsEl = document.getElementById('trainingSteps');
  const outputBiasEl = document.getElementById('outputBias');

  const trendCanvas = document.getElementById('trend');
  const tctx = trendCanvas.getContext('2d');
  const weightsGrid = document.getElementById('weightsGrid');
  const outputWeightsCanvas = document.getElementById('outputWeights');
  const owctx = outputWeightsCanvas.getContext('2d');

  const startBtn = document.getElementById('startBtn');
  const pauseBtn = document.getElementById('pauseBtn');
  const resetBtn = document.getElementById('resetBtn');

  const lrEl = document.getElementById('learningRate');
  const lrVal = document.getElementById('learningRateVal');
  const hiddenEl = document.getElementById('hiddenUnits');
  const hiddenVal = document.getElementById('hiddenUnitsVal');
  const epsEl = document.getElementById('epsilon');
  const epsVal = document.getElementById('epsilonVal');
  const delayEl = document.getElementById('delayMs');
  const delayVal = document.getElementById('delayMsVal');

  const worker = new Worker('worker.js');

  let confidenceHistory = [];
  const maxTrendPoints = 240;
  let running = false;

  function drawBoard() {
    const W = cvs.width, H = cvs.height;
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle = '#f1d6a0';
    ctx.fillRect(0,0,W,H);

    const margin = 24;
    const n = size;
    const cell = (W - margin*2) / (n - 1);
    ctx.strokeStyle = '#111';
    ctx.lineWidth = 1;

    for (let i = 0; i < n; i++) {
      const x = margin + i * cell;
      ctx.beginPath();
      ctx.moveTo(margin, margin + i * cell);
      ctx.lineTo(W - margin, margin + i * cell);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(x, margin);
      ctx.lineTo(x, H - margin);
      ctx.stroke();
    }

    const stars = [[2,2],[6,2],[4,4],[2,6],[6,6]];
    ctx.fillStyle = '#111';
    for (const [sx,sy] of stars) {
      const px = margin + sx * cell;
      const py = margin + sy * cell;
      ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI*2); ctx.fill();
    }

    for (let i = 0; i < n*n; i++) {
      const v = board.cells[i];
      if (v === 0) continue;
      const [x,y] = xyOf(i, n);
      const px = margin + x * cell;
      const py = margin + y * cell;
      ctx.beginPath();
      ctx.arc(px, py, cell*0.45, 0, Math.PI*2);
      ctx.fillStyle = (v === BLACK) ? '#111' : '#fff';
      ctx.fill();
      ctx.strokeStyle = '#111';
      ctx.stroke();
      if (v === WHITE) {
        ctx.beginPath();
        ctx.arc(px - cell*0.12, py - cell*0.12, cell*0.4, 0, Math.PI*2);
        ctx.strokeStyle = '#ccc';
        ctx.stroke();
      }
    }

    if (lastMove != null && lastMove >= 0) {
      const [lx, ly] = xyOf(lastMove, n);
      const px = margin + lx * cell;
      const py = margin + ly * cell;
      ctx.strokeStyle = '#f25f1c';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px, py, cell*0.2, 0, Math.PI*2);
      ctx.stroke();
    }
  }

  function applyBoardSnapshot(snap) {
    if (!snap) return;
    board.cells = new Uint8Array(snap.cells || board.cells);
    board.toPlay = snap.toPlay ?? board.toPlay;
    board.capturesB = snap.capturesB ?? board.capturesB;
    board.capturesW = snap.capturesW ?? board.capturesW;
    board.moveCount = snap.moveCount ?? board.moveCount;
    board.passes = snap.passes ?? board.passes;
    lastMove = snap.lastMove ?? -1;
    drawBoard();
    updateHUD();
  }

  function updateHUD() {
    turnLabel.textContent = board.toPlay === BLACK ? 'Black' : 'White';
    moveCountEl.textContent = board.moveCount ?? 0;
    capBEl.textContent = board.capturesB ?? 0;
    capWEl.textContent = board.capturesW ?? 0;
  }

  function setStatus(text) {
    if (text) statusText.textContent = text;
    else statusText.textContent = running ? 'Running' : 'Idle';
  }

  function setConfidence(val) {
    if (val == null) {
      confNowEl.textContent = '—';
      return;
    }
    const pct = (val * 100).toFixed(1);
    confNowEl.textContent = pct + '%';
  }

  function pushConfidence(val) {
    if (val == null || Number.isNaN(val)) return;
    confidenceHistory.push(val);
    if (confidenceHistory.length > maxTrendPoints) {
      confidenceHistory = confidenceHistory.slice(confidenceHistory.length - maxTrendPoints);
    }
    drawTrend();
  }

  function resetTrend() {
    confidenceHistory = [];
    drawTrend();
  }

  function drawTrend() {
    const W = trendCanvas.width;
    const H = trendCanvas.height;
    tctx.clearRect(0,0,W,H);
    tctx.fillStyle = '#fff';
    tctx.fillRect(0,0,W,H);
    tctx.strokeStyle = '#000';
    tctx.strokeRect(0,0,W,H);

    tctx.strokeStyle = '#aaa';
    tctx.beginPath();
    tctx.moveTo(0, H * 0.5);
    tctx.lineTo(W, H * 0.5);
    tctx.stroke();

    if (!confidenceHistory.length) return;
    tctx.strokeStyle = '#f27405';
    tctx.beginPath();
    const stepX = confidenceHistory.length > 1 ? W / (confidenceHistory.length - 1) : W;
    confidenceHistory.forEach((v, idx) => {
      const x = idx * stepX;
      const y = H - v * H;
      if (idx === 0) tctx.moveTo(x, y); else tctx.lineTo(x, y);
    });
    tctx.stroke();
  }

  function colorForValue(v, maxAbs) {
    if (maxAbs <= 1e-6) return 'rgb(240,240,240)';
    const ratio = v / maxAbs;
    const abs = Math.min(1, Math.abs(ratio));
    const hue = ratio >= 0 ? 25 : 210; // orange vs blue
    const sat = 80;
    const light = 55 - abs * 25;
    return `hsl(${hue}, ${sat}%, ${light}%)`;
  }

  function drawWeightCanvas(canvas, weights, boardSize) {
    const ctxW = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    ctxW.clearRect(0,0,W,H);
    ctxW.fillStyle = '#fff';
    ctxW.fillRect(0,0,W,H);
    const margin = 6;
    const n = boardSize;
    const cellW = (W - margin*2) / n;
    const cellH = (H - margin*2) / n;
    let maxAbs = 0;
    for (const w of weights) maxAbs = Math.max(maxAbs, Math.abs(w));
    if (maxAbs <= 1e-6) maxAbs = 1;

    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const idx = y * n + x;
        const v = weights[idx] || 0;
        ctxW.fillStyle = colorForValue(v, maxAbs);
        ctxW.fillRect(margin + x * cellW, margin + y * cellH, cellW, cellH);
      }
    }

    ctxW.strokeStyle = 'rgba(0,0,0,0.15)';
    ctxW.lineWidth = 1;
    for (let i = 0; i <= n; i++) {
      const x = margin + i * cellW;
      ctxW.beginPath(); ctxW.moveTo(x, margin); ctxW.lineTo(x, margin + n * cellH); ctxW.stroke();
      const y = margin + i * cellH;
      ctxW.beginPath(); ctxW.moveTo(margin, y); ctxW.lineTo(margin + n * cellW, y); ctxW.stroke();
    }
  }

  function drawOutputWeights(weights) {
    const W = outputWeightsCanvas.width;
    const H = outputWeightsCanvas.height;
    owctx.clearRect(0,0,W,H);
    owctx.fillStyle = '#fff';
    owctx.fillRect(0,0,W,H);
    owctx.strokeStyle = '#000';
    owctx.strokeRect(0,0,W,H);

    owctx.strokeStyle = '#aaa';
    owctx.beginPath();
    owctx.moveTo(0, H * 0.5);
    owctx.lineTo(W, H * 0.5);
    owctx.stroke();

    if (!weights || !weights.length) return;
    let maxAbs = 0;
    for (const w of weights) maxAbs = Math.max(maxAbs, Math.abs(w));
    if (maxAbs <= 1e-6) maxAbs = 1;
    const barWidth = W / weights.length;
    const usableHeight = H * 0.8;
    for (let i = 0; i < weights.length; i++) {
      const v = weights[i];
      const ratio = v / maxAbs;
      const barH = Math.abs(ratio) * usableHeight * 0.5;
      const x = i * barWidth + barWidth * 0.15;
      const y = v >= 0 ? (H * 0.5 - barH) : H * 0.5;
      owctx.fillStyle = colorForValue(v, maxAbs);
      owctx.fillRect(x, y, barWidth * 0.7, Math.max(2, barH));
    }
  }

  function renderWeights(data) {
    weightsGrid.innerHTML = '';
    if (!data) {
      drawOutputWeights([]);
      outputBiasEl.textContent = '0.000';
      return;
    }
    const boardCells = data.boardSize * data.boardSize;
    for (let i = 0; i < data.hiddenUnits; i++) {
      const start = i * data.inputSize;
      const slice = data.w1.slice(start, start + boardCells);
      const toPlayWeight = data.w1[start + boardCells] || 0;
      const unit = document.createElement('div');
      unit.className = 'weight-unit';
      const canvas = document.createElement('canvas');
      canvas.width = 140;
      canvas.height = 140;
      unit.appendChild(canvas);
      drawWeightCanvas(canvas, slice, data.boardSize);
      const caption = document.createElement('div');
      caption.className = 'weight-caption';
      const bias = data.b1[i] ?? 0;
      const outW = data.w2[i] ?? 0;
      caption.textContent = `h${i+1}: bias=${bias.toFixed(3)} · out=${outW.toFixed(3)} · toPlay=${toPlayWeight.toFixed(3)}`;
      unit.appendChild(caption);
      weightsGrid.appendChild(unit);
    }
    drawOutputWeights(data.w2 || []);
    if (typeof data.b2 === 'number') outputBiasEl.textContent = data.b2.toFixed(3);
  }

  function updateStats(stats) {
    if (!stats) return;
    totalGamesEl.textContent = stats.games ?? 0;
    blackWinsEl.textContent = stats.blackWins ?? 0;
    whiteWinsEl.textContent = stats.whiteWins ?? 0;
    if (stats.games > 0) {
      blackWinRateEl.textContent = (stats.blackWinRate * 100).toFixed(1) + '%';
    } else {
      blackWinRateEl.textContent = '—';
    }
    if (stats.predictionAccuracy != null && stats.totalPredictions > 0) {
      predAccuracyEl.textContent = (stats.predictionAccuracy * 100).toFixed(1) + '%';
    } else {
      predAccuracyEl.textContent = '—';
    }
    if (stats.avgConfidence != null && stats.games > 0) {
      avgConfidenceEl.textContent = (stats.avgConfidence * 100).toFixed(1) + '%';
    } else {
      avgConfidenceEl.textContent = '—';
    }
    trainingStepsEl.textContent = stats.trainingSteps ?? 0;
  }

  function updateLastGame(info) {
    if (!info) return;
    gameCountEl.textContent = info.gameNumber ?? 0;
    lastWinnerEl.textContent = info.winner ?? '—';
    if (typeof info.score === 'number') {
      lastScoreEl.textContent = info.score.toFixed(1);
    } else {
      lastScoreEl.textContent = '—';
    }
  }

  function applyConfig(cfg) {
    if (!cfg) return;
    if (cfg.learningRate != null) {
      lrEl.value = cfg.learningRate.toFixed(2);
      lrVal.textContent = (+cfg.learningRate).toFixed(2);
    }
    if (cfg.hiddenUnits != null) {
      hiddenEl.value = cfg.hiddenUnits;
      hiddenVal.textContent = cfg.hiddenUnits;
    }
    if (cfg.epsilon != null) {
      epsEl.value = cfg.epsilon.toFixed(2);
      epsVal.textContent = (+cfg.epsilon).toFixed(2);
    }
    if (cfg.delayMs != null) {
      delayEl.value = cfg.delayMs;
      delayVal.textContent = cfg.delayMs;
    }
  }

  function setRunningState(isRunning) {
    running = isRunning;
    startBtn.disabled = isRunning;
    pauseBtn.disabled = !isRunning;
    setStatus();
  }

  worker.onmessage = (ev) => {
    const msg = ev.data || {};
    if (msg.type === 'init' || msg.type === 'resetDone') {
      applyBoardSnapshot(msg.board);
      setConfidence(msg.confidence);
      resetTrend();
      if (msg.confidence != null) pushConfidence(msg.confidence);
      updateStats(msg.stats);
      if (msg.stats) {
        const info = {
          gameNumber: msg.stats.games,
          winner: msg.stats.lastWinner,
          score: msg.stats.lastScore
        };
        updateLastGame(info);
      }
      renderWeights(msg.weights);
      applyConfig(msg.config);
      setRunningState(false);
    } else if (msg.type === 'status') {
      setRunningState(!!msg.running);
    } else if (msg.type === 'gameStart') {
      applyBoardSnapshot(msg.board);
      setConfidence(msg.confidence);
      if (msg.resetTrend) resetTrend();
      setStatus(`Game ${msg.gameNumber} — starting`);
    } else if (msg.type === 'move') {
      applyBoardSnapshot(msg.board);
      setConfidence(msg.confidence);
      pushConfidence(msg.confidence);
      const mv = msg.lastMove;
      if (mv != null && mv >= 0) {
        const [x,y] = xyOf(mv, size);
        setStatus(`Game ${msg.gameNumber} — Move ${msg.moveIndex} (${msg.lastPlayer === BLACK ? 'B' : 'W'} ${x+1},${y+1})`);
      } else {
        setStatus(`Game ${msg.gameNumber} — Move ${msg.moveIndex} (pass)`);
      }
    } else if (msg.type === 'gameComplete') {
      updateStats(msg.stats);
      updateLastGame({ gameNumber: msg.stats?.games, winner: msg.winner, score: msg.score });
      renderWeights(msg.weights);
      setStatus(`Game ${msg.stats?.games} finished — ${msg.winner} by ${(msg.score ?? 0).toFixed(1)}`);
    } else if (msg.type === 'weights') {
      renderWeights(msg.weights);
    } else if (msg.type === 'stats') {
      updateStats(msg.stats);
      if (msg.stats) {
        const info = {
          gameNumber: msg.stats.games,
          winner: msg.stats.lastWinner,
          score: msg.stats.lastScore
        };
        updateLastGame(info);
      }
      if (msg.weights) renderWeights(msg.weights);
    } else if (msg.type === 'config') {
      applyConfig(msg.config);
    }
  };

  startBtn.addEventListener('click', () => {
    worker.postMessage({ type: 'start' });
  });

  pauseBtn.addEventListener('click', () => {
    worker.postMessage({ type: 'pause' });
  });

  resetBtn.addEventListener('click', () => {
    worker.postMessage({ type: 'reset' });
  });

  function handleSlider(el, valEl, formatter, key) {
    el.addEventListener('input', () => {
      valEl.textContent = formatter(el.value);
    });
    el.addEventListener('change', () => {
      valEl.textContent = formatter(el.value);
      const payload = { type: 'configure', config: { [key]: parseFloat(el.value) } };
      worker.postMessage(payload);
    });
  }

  handleSlider(lrEl, lrVal, v => (+v).toFixed(2), 'learningRate');
  handleSlider(hiddenEl, hiddenVal, v => Math.round(+v), 'hiddenUnits');
  handleSlider(epsEl, epsVal, v => (+v).toFixed(2), 'epsilon');
  handleSlider(delayEl, delayVal, v => Math.round(+v), 'delayMs');

  // Kick initial render with blank board
  drawBoard();
  updateHUD();
  setStatus('Idle');
})();
