// UI orchestration for Go board + MCTS stats.
(() => {
  const { Board, BLACK, WHITE, other, idxOf, xyOf } = window.GoEngine;

  // --- DOM refs
  const cvs = document.getElementById('board');
  const ctx = cvs.getContext('2d');

  const capBEl = document.getElementById('capB');
  const capWEl = document.getElementById('capW');
  const turnLabel = document.getElementById('turnLabel');
  const statusText = document.getElementById('statusText');

  const newGameBtn = document.getElementById('newGame');
  const passBtn = document.getElementById('passBtn');
  const undoBtn = document.getElementById('undoBtn');
  const aiMoveBtn = document.getElementById('aiMove');

  const timeMsEl = document.getElementById('timeMs');
  const timeMsVal = document.getElementById('timeMsVal');
  const cExpEl = document.getElementById('cExp');
  const cExpVal = document.getElementById('cExpVal');
  const komiLabel = document.getElementById('komiLabel');

  const aiPlaysBlack = document.getElementById('aiPlaysBlack');
  const aiPlaysWhite = document.getElementById('aiPlaysWhite');
  const autoAI = document.getElementById('autoAI');

  const itCountEl = document.getElementById('itCount');
  const ppsEl = document.getElementById('pps');
  const nodesEl = document.getElementById('nodes');
  const avgDepthEl = document.getElementById('avgDepth');
  const winrateEl = document.getElementById('winrate');
  const bestMoveEl = document.getElementById('bestMove');

  const heatmap = document.getElementById('heatmap');
  const hctx = heatmap.getContext('2d');
  const trend = document.getElementById('trend');
  const tctx = trend.getContext('2d');

  // --- State
  const size = 9;
  const komi = 6.5;
  const board = new Board(size, komi);
  komiLabel.textContent = komi.toString();

  // Worker
  const worker = new Worker('worker.js');

  let searching = false;
  let lastHeat = new Array(size*size).fill(0);
  let trendVals = [];

  // --- Drawing
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

    // Grid
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

    // Star points for 9x9: (2,2), (6,2), (4,4), (2,6), (6,6)
    const stars = [[2,2],[6,2],[4,4],[2,6],[6,6]];
    ctx.fillStyle = '#111';
    for (const [sx,sy] of stars) {
      const px = margin + sx * cell;
      const py = margin + sy * cell;
      ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI*2); ctx.fill();
    }

    // Stones
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
        // little shadow line
        ctx.beginPath();
        ctx.arc(px - cell*0.12, py - cell*0.12, cell*0.4, 0, Math.PI*2);
        ctx.strokeStyle = '#ccc'; ctx.stroke();
      }
    }

    // Ko mark
    if (board.ko >= 0) {
      const [kx,ky] = xyOf(board.ko, n);
      const px = margin + kx * cell;
      const py = margin + ky * cell;
      ctx.fillStyle = 'rgba(255,0,0,0.6)';
      ctx.fillRect(px-3, py-3, 6, 6);
    }
  }

  function drawHeatmap(values) {
    const W = heatmap.width, H = heatmap.height;
    const margin = 6;
    const n = size;
    const cell = (W - margin*2) / n;
    const max = Math.max(1, ...values);
    hctx.clearRect(0,0,W,H);
    hctx.fillStyle = '#fff'; hctx.fillRect(0,0,W,H);
    for (let y=0; y<n; y++) {
      for (let x=0; x<n; x++) {
        const i = y*n + x;
        const v = values[i] / max; // 0..1
        const shade = 255 - Math.floor(v * 255);
        hctx.fillStyle = `rgb(${shade},${shade},${shade})`;
        hctx.fillRect(margin + x*cell, margin + y*cell, cell-1, cell-1);
      }
    }
    // Grid lines
    hctx.strokeStyle = '#000';
    for (let i=0;i<=n;i++) {
      const x = margin + i*cell;
      hctx.beginPath(); hctx.moveTo(x, margin); hctx.lineTo(x, margin + n*cell); hctx.stroke();
      const y = margin + i*cell;
      hctx.beginPath(); hctx.moveTo(margin, y); hctx.lineTo(margin + n*cell, y); hctx.stroke();
    }
  }

  function drawTrend(vals) {
    const W = trend.width, H = trend.height;
    tctx.clearRect(0,0,W,H);
    tctx.fillStyle = '#fff'; tctx.fillRect(0,0,W,H);
    tctx.strokeStyle = '#000'; tctx.strokeRect(0,0,W,H);

    if (!vals.length) return;
    const maxN = Math.min(vals.length, 200);
    const slice = vals.slice(vals.length - maxN);
    const stepX = W / (slice.length - 1 || 1);
    tctx.beginPath();
    for (let i=0;i<slice.length;i++) {
      const x = i * stepX;
      const y = H - slice[i] * H;
      if (i===0) tctx.moveTo(x,y); else tctx.lineTo(x,y);
    }
    tctx.stroke();
    // baseline 0.5
    tctx.strokeStyle = '#aaa';
    tctx.beginPath(); tctx.moveTo(0, H*0.5); tctx.lineTo(W, H*0.5); tctx.stroke();
  }

  function updateHUD(status='') {
    turnLabel.textContent = (board.toPlay === BLACK) ? 'Black' : 'White';
    capBEl.textContent = board.capturesB;
    capWEl.textContent = board.capturesW;
    statusText.textContent = status || statusText.textContent;
  }

  // --- Interaction
  function canvasToIdx(evt) {
    const rect = cvs.getBoundingClientRect();
    const mx = (evt.clientX - rect.left) * (cvs.width / rect.width);
    const my = (evt.clientY - rect.top) * (cvs.height / rect.height);
    const margin = 24;
    const cell = (cvs.width - margin*2) / (size - 1);
    const x = Math.round((mx - margin) / cell);
    const y = Math.round((my - margin) / cell);
    if (x < 0 || x >= size || y < 0 || y >= size) return -1;
    return y*size + x;
  }

  cvs.addEventListener('click', (e) => {
    if (searching) return;
    const i = canvasToIdx(e);
    if (i < 0) return;
    if (isAIToMove()) return;
    const res = board.play(i);
    if (!res.ok) { updateHUD(`Illegal: ${res.msg}`); return; }
    afterHumanMove();
  });

  function isAIToMove() {
    return (board.toPlay === BLACK && aiPlaysBlack.checked) ||
           (board.toPlay === WHITE && aiPlaysWhite.checked);
  }

  function afterHumanMove() {
    drawBoard();
    updateHUD('');
    if (autoAI.checked && isAIToMove()) {
      thinkAndMove();
    }
  }

  newGameBtn.addEventListener('click', () => {
    const b = new Board(size, komi);
    board.cells = b.cells;
    board.toPlay = b.toPlay;
    board.ko = -1;
    board.passes = 0;
    board.moveCount = 0;
    board.capturesB = 0; board.capturesW = 0;
    board.history = [];
    lastHeat = new Array(size*size).fill(0);
    trendVals = [];
    updateStats({it:0,nodes:0,pps:0,avgDepth:0,winrate:null,bestMove:null,heat:lastHeat,trend:trendVals});
    drawBoard(); drawHeatmap(lastHeat); drawTrend(trendVals);
    updateHUD('New game.');
  });

  passBtn.addEventListener('click', () => {
    if (searching) return;
    const r = board.play(-1);
    if (!r.ok) return;
    if (board.isTerminal()) {
      endGame();
    } else {
      afterHumanMove();
    }
  });

  undoBtn.addEventListener('click', () => {
    if (searching) return;
    board.undo(); board.undo(); // undo both last moves if possible
    drawBoard(); updateHUD('Undo');
  });

  aiMoveBtn.addEventListener('click', () => {
    if (!isAIToMove()) { updateHUD('It is not the AI side to move.'); return; }
    if (searching) return;
    thinkAndMove();
  });

  timeMsEl.addEventListener('input', () => timeMsVal.textContent = timeMsEl.value);
  cExpEl.addEventListener('input', () => cExpVal.textContent = (+cExpEl.value).toFixed(2));

  // --- Search
  worker.onmessage = (ev) => {
    const msg = ev.data || {};
    if (msg.type === 'progress') {
      updateStats(msg.snap);
    } else if (msg.type === 'done') {
      searching = false;
      updateStats(msg.res);
      const mv = msg.res.move ?? -1;
      board.play(mv ?? -1);
      if (board.isTerminal()) { endGame(); return; }
      drawBoard();
      updateHUD('AI played.');
      if (autoAI.checked && isAIToMove()) {
        // If both sides are AI, chain
        thinkAndMove();
      }
    }
  };

  function updateStats(s) {
    itCountEl.textContent = s.it ?? 0;
    ppsEl.textContent = s.pps ?? 0;
    nodesEl.textContent = s.nodes ?? 0;
    avgDepthEl.textContent = s.avgDepth ?? 0;
    if (s.winrate != null) winrateEl.textContent = (s.winrate*100).toFixed(1) + '%';
    if (s.bestMove != null && s.bestMove >= 0) {
      const [x,y] = xyOf(s.bestMove, size);
      bestMoveEl.textContent = `(${x+1}, ${y+1})`;
    } else {
      bestMoveEl.textContent = 'pass';
    }
    if (s.heat) { lastHeat = s.heat.slice(); drawHeatmap(lastHeat); }
    if (s.trend) { trendVals = s.trend.slice(); drawTrend(trendVals); }
  }

  function thinkAndMove() {
    searching = true;
    updateHUD('AI thinking...');
    const payload = {
      type: 'search',
      state: {
        size, komi,
        cells: Array.from(board.cells),
        toPlay: board.toPlay,
        ko: board.ko,
        passes: board.passes,
        moveCount: board.moveCount
      },
      config: {
        timeMs: +timeMsEl.value,
        c: +cExpEl.value
      }
    };
    worker.postMessage(payload);
  }

  function endGame() {
    const score = board.areaScore();
    const winner = (score > 0) ? 'Black' : 'White';
    drawBoard();
    updateHUD(`Game over. Score (B-W): ${score.toFixed(1)}. Winner: ${winner}`);
  }

  // Initial render
  timeMsVal.textContent = timeMsEl.value;
  cExpVal.textContent = (+cExpEl.value).toFixed(2);
  drawBoard(); drawHeatmap(lastHeat); drawTrend(trendVals);
  updateHUD('Ready.');

  // If AI starts as black, auto-move
  if (isAIToMove() && autoAI.checked) thinkAndMove();
})();