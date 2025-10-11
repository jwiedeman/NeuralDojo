importScripts('go_engine.js');

const { Board, BLACK, WHITE } = self.GoEngine;

class TinyValueNet {
  constructor(boardSize, hiddenUnits = 6, learningRate = 0.1) {
    this.boardSize = boardSize;
    this.inputSize = boardSize * boardSize + 1; // board + player feature
    this.hiddenUnits = hiddenUnits;
    this.learningRate = learningRate;
    this.initWeights();
  }

  initWeights() {
    const scale1 = 0.2;
    this.w1 = new Float32Array(this.hiddenUnits * this.inputSize);
    this.b1 = new Float32Array(this.hiddenUnits);
    for (let i = 0; i < this.w1.length; i++) {
      this.w1[i] = (Math.random() * 2 - 1) * scale1;
    }
    this.b1.fill(0);
    this.w2 = new Float32Array(this.hiddenUnits);
    const scale2 = 0.2;
    for (let i = 0; i < this.w2.length; i++) {
      this.w2[i] = (Math.random() * 2 - 1) * scale2;
    }
    this.b2 = 0;
  }

  setLearningRate(lr) {
    this.learningRate = lr;
  }

  setHiddenUnits(units) {
    this.hiddenUnits = units;
    this.initWeights();
  }

  forward(features) {
    const hidden = new Float32Array(this.hiddenUnits);
    const z1 = new Float32Array(this.hiddenUnits);
    for (let h = 0; h < this.hiddenUnits; h++) {
      let sum = this.b1[h];
      const offset = h * this.inputSize;
      for (let i = 0; i < this.inputSize; i++) {
        sum += this.w1[offset + i] * features[i];
      }
      z1[h] = sum;
      hidden[h] = Math.tanh(sum);
    }
    let z2 = this.b2;
    for (let h = 0; h < this.hiddenUnits; h++) {
      z2 += this.w2[h] * hidden[h];
    }
    const clipped = Math.max(-10, Math.min(10, z2));
    const output = 1 / (1 + Math.exp(-clipped));
    return { hidden, z1, z2: clipped, output };
  }

  trainSample(features, target) {
    const { hidden, output } = this.forward(features);
    const lr = this.learningRate;
    const error = output - target;
    const dOut = error * output * (1 - output);

    const gradHidden = new Float32Array(this.hiddenUnits);
    for (let h = 0; h < this.hiddenUnits; h++) {
      gradHidden[h] = dOut * this.w2[h] * (1 - hidden[h] * hidden[h]);
    }

    for (let h = 0; h < this.hiddenUnits; h++) {
      const deltaW2 = dOut * hidden[h];
      this.w2[h] -= lr * deltaW2;
    }
    this.b2 -= lr * dOut;

    for (let h = 0; h < this.hiddenUnits; h++) {
      const offset = h * this.inputSize;
      const grad = gradHidden[h];
      for (let i = 0; i < this.inputSize; i++) {
        this.w1[offset + i] -= lr * grad * features[i];
      }
      this.b1[h] -= lr * grad;
    }
  }

  trainBatch(samples, target) {
    for (const feat of samples) {
      this.trainSample(feat, target);
    }
  }
}

const config = {
  size: 9,
  komi: 6.5,
  hiddenUnits: 6,
  learningRate: 0.1,
  epsilon: 0.1,
  delayMs: 160
};

let net = new TinyValueNet(config.size, config.hiddenUnits, config.learningRate);
let running = false;
let moveTimer = null;
let board = new Board(config.size, config.komi);
let states = [];
let predictions = [];
let stats = createStats();

function createStats() {
  return {
    games: 0,
    blackWins: 0,
    whiteWins: 0,
    totalPredictions: 0,
    correctPredictions: 0,
    avgConfidenceSum: 0,
    lastWinner: '—',
    lastScore: 0,
    trainingSteps: 0
  };
}

function encodeBoard(b) {
  const arr = new Float32Array(config.size * config.size + 1);
  const total = config.size * config.size;
  for (let i = 0; i < total; i++) {
    const v = b.cells[i];
    if (v === BLACK) arr[i] = 1;
    else if (v === WHITE) arr[i] = -1;
    else arr[i] = 0;
  }
  arr[arr.length - 1] = b.toPlay === BLACK ? 1 : -1;
  return arr;
}

function snapshotBoard() {
  return {
    cells: Array.from(board.cells),
    toPlay: board.toPlay,
    capturesB: board.capturesB,
    capturesW: board.capturesW,
    moveCount: board.moveCount,
    passes: board.passes,
    lastMove
  };
}

let lastMove = -1;
let currentGame = 1;

function clearTimer() {
  if (moveTimer != null) {
    clearTimeout(moveTimer);
    moveTimer = null;
  }
}

function setRunning(val) {
  running = val;
  self.postMessage({ type: 'status', running });
  if (!running) {
    clearTimer();
  } else {
    scheduleNext();
  }
}

function scheduleNext() {
  clearTimer();
  if (!running) return;
  if (board.isTerminal()) {
    finishGame();
    return;
  }
  const delay = Math.max(0, config.delayMs | 0);
  moveTimer = setTimeout(stepSelfPlay, delay);
}

function stepSelfPlay() {
  moveTimer = null;
  if (!running) return;
  if (board.isTerminal()) {
    finishGame();
    return;
  }

  const encoded = encodeBoard(board);
  const { output } = net.forward(encoded);
  states.push(encoded);
  predictions.push(output);

  const move = chooseMove();
  const mover = board.toPlay;
  lastMove = move;
  board.play(move);

  self.postMessage({
    type: 'move',
    board: snapshotBoard(),
    confidence: output,
    gameNumber: currentGame,
    moveIndex: board.moveCount,
    lastMove: move,
    lastPlayer: mover
  });

  scheduleNext();
}

function chooseMove() {
  const moves = board.legalMoves(true);
  if (!moves.length) return -1;
  if (Math.random() < config.epsilon) {
    return moves[(Math.random() * moves.length) | 0];
  }
  let bestMove = moves[0];
  let bestValue = board.toPlay === BLACK ? -Infinity : Infinity;
  for (const mv of moves) {
    const clone = board.clone();
    const res = clone.play(mv);
    if (!res.ok) continue;
    const { output } = net.forward(encodeBoard(clone));
    if (board.toPlay === BLACK) {
      if (output > bestValue) {
        bestValue = output;
        bestMove = mv;
      }
    } else {
      if (output < bestValue) {
        bestValue = output;
        bestMove = mv;
      }
    }
  }
  return bestMove;
}

function finishGame() {
  const score = board.areaScore();
  const blackWin = score > 0 ? 1 : 0;
  if (states.length) {
    net.trainBatch(states, blackWin);
    stats.trainingSteps += states.length;
  }
  const correct = predictions.reduce((acc, p) => acc + (((p >= 0.5) ? 1 : 0) === blackWin ? 1 : 0), 0);
  stats.games += 1;
  if (blackWin) stats.blackWins += 1; else stats.whiteWins += 1;
  stats.totalPredictions += predictions.length;
  stats.correctPredictions += correct;
  const avgConf = predictions.length ? predictions.reduce((a,b) => a + b, 0) / predictions.length : 0.5;
  stats.avgConfidenceSum += avgConf;
  stats.lastWinner = blackWin ? 'Black' : 'White';
  stats.lastScore = score;

  self.postMessage({
    type: 'gameComplete',
    stats: formatStats(),
    winner: stats.lastWinner,
    score,
    weights: exportWeights()
  });

  currentGame = stats.games + 1;
  startNewGame();
}

function startNewGame() {
  clearTimer();
  board = new Board(config.size, config.komi);
  states = [];
  predictions = [];
  lastMove = -1;
  const { output } = net.forward(encodeBoard(board));
  self.postMessage({
    type: 'gameStart',
    board: snapshotBoard(),
    confidence: output,
    gameNumber: currentGame,
    running
  });
  if (running) scheduleNext();
}

function exportWeights() {
  return {
    hiddenUnits: net.hiddenUnits,
    inputSize: net.inputSize,
    boardSize: net.boardSize,
    w1: Array.from(net.w1),
    b1: Array.from(net.b1),
    w2: Array.from(net.w2),
    b2: net.b2
  };
}

function formatStats() {
  const res = {
    games: stats.games,
    blackWins: stats.blackWins,
    whiteWins: stats.whiteWins,
    blackWinRate: stats.games ? stats.blackWins / stats.games : 0,
    predictionAccuracy: stats.totalPredictions ? stats.correctPredictions / stats.totalPredictions : 0,
    totalPredictions: stats.totalPredictions,
    avgConfidence: stats.games ? stats.avgConfidenceSum / stats.games : 0,
    trainingSteps: stats.trainingSteps,
    lastWinner: stats.lastWinner,
    lastScore: stats.lastScore
  };
  return res;
}

function fullReset(messageType = 'resetDone') {
  clearTimer();
  net = new TinyValueNet(config.size, config.hiddenUnits, config.learningRate);
  stats = createStats();
  board = new Board(config.size, config.komi);
  states = [];
  predictions = [];
  lastMove = -1;
  currentGame = 1;
  const { output } = net.forward(encodeBoard(board));
  self.postMessage({
    type: messageType,
    board: snapshotBoard(),
    stats: formatStats(),
    weights: exportWeights(),
    confidence: output,
    config: { ...config },
    running
  });
}

self.onmessage = (ev) => {
  const data = ev.data || {};
  if (data.type === 'start') {
    setRunning(true);
  } else if (data.type === 'pause') {
    setRunning(false);
  } else if (data.type === 'reset') {
    setRunning(false);
    fullReset('resetDone');
  } else if (data.type === 'configure') {
    const cfg = data.config || {};
    let needsReset = false;
    if (cfg.learningRate != null) {
      config.learningRate = +cfg.learningRate;
      net.setLearningRate(config.learningRate);
    }
    if (cfg.hiddenUnits != null) {
      const units = Math.max(2, Math.floor(+cfg.hiddenUnits));
      if (units !== config.hiddenUnits) {
        config.hiddenUnits = units;
        needsReset = true;
      }
    }
    if (cfg.epsilon != null) {
      config.epsilon = Math.max(0, Math.min(0.5, +cfg.epsilon));
    }
    if (cfg.delayMs != null) {
      config.delayMs = Math.max(0, Math.floor(+cfg.delayMs));
    }
    if (needsReset) {
      fullReset('resetDone');
    } else {
      self.postMessage({ type: 'config', config: { ...config } });
      self.postMessage({ type: 'weights', weights: exportWeights() });
    }
  }
};

fullReset('init');
startNewGame();
