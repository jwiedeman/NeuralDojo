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
  delayMs: 160,
  captureBias: 0.15
};

function createNet() {
  return new TinyValueNet(config.size, config.hiddenUnits, config.learningRate);
}

let nets = {
  black: createNet(),
  white: createNet()
};
let running = false;
let moveTimer = null;
let board = new Board(config.size, config.komi);
let statesByColor = { black: [], white: [] };
let predictionsByColor = { black: [], white: [] };
let stats = createStats();
let gameHistory = [];
const maxHistory = 360;

function createStats() {
  return {
    games: 0,
    blackWins: 0,
    whiteWins: 0,
    totalPredictions: 0,
    correctPredictions: 0,
    confidenceSum: 0,
    lastWinner: '—',
    lastScore: 0,
    trainingSteps: 0,
    perModel: {
      black: createModelStats(),
      white: createModelStats()
    }
  };
}

function createModelStats() {
  return {
    predictions: 0,
    correctPredictions: 0,
    confidenceSum: 0,
    trainingSteps: 0
  };
}

function encodeBoardFor(b, perspective) {
  const arr = new Float32Array(config.size * config.size + 1);
  const total = config.size * config.size;
  const opp = perspective === BLACK ? WHITE : BLACK;
  for (let i = 0; i < total; i++) {
    const v = b.cells[i];
    if (v === perspective) arr[i] = 1;
    else if (v === opp) arr[i] = -1;
    else arr[i] = 0;
  }
  arr[arr.length - 1] = b.toPlay === perspective ? 1 : -1;
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

function colorKey(player) {
  return player === BLACK ? 'black' : 'white';
}

function evaluateBoardConfidence(b) {
  return {
    black: nets.black.forward(encodeBoardFor(b, BLACK)).output,
    white: nets.white.forward(encodeBoardFor(b, WHITE)).output
  };
}

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

  const encoded = encodeBoardFor(board, board.toPlay);
  const playerKey = colorKey(board.toPlay);
  const mover = board.toPlay;
  const net = nets[playerKey];
  const { output } = net.forward(encoded);
  statesByColor[playerKey].push(encoded);
  predictionsByColor[playerKey].push(output);

  const move = chooseMove();
  lastMove = move;
  board.play(move);

  const confidence = evaluateBoardConfidence(board);

  self.postMessage({
    type: 'move',
    board: snapshotBoard(),
    confidence,
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
  const playerKey = colorKey(board.toPlay);
  const net = nets[playerKey];
  const mover = board.toPlay;
  let bestMove = moves[0];
  let bestValue = -Infinity;
  for (const mv of moves) {
    const clone = board.clone();
    const res = clone.play(mv);
    if (!res.ok) continue;
    const { output } = net.forward(encodeBoardFor(clone, mover));
    let value = output;
    if (config.captureBias > 0 && res.captured) {
      value += config.captureBias * res.captured;
    }
    if (value > bestValue) {
      bestValue = value;
      bestMove = mv;
    }
  }
  return bestMove;
}

function finishGame() {
  const score = board.areaScore();
  const blackWin = score > 0 ? 1 : 0;
  const whiteWin = blackWin ? 0 : 1;

  const blackStates = statesByColor.black;
  const whiteStates = statesByColor.white;
  if (blackStates.length) {
    nets.black.trainBatch(blackStates, blackWin);
    stats.trainingSteps += blackStates.length;
    stats.perModel.black.trainingSteps += blackStates.length;
  }
  if (whiteStates.length) {
    nets.white.trainBatch(whiteStates, whiteWin);
    stats.trainingSteps += whiteStates.length;
    stats.perModel.white.trainingSteps += whiteStates.length;
  }

  const blackPreds = predictionsByColor.black;
  const whitePreds = predictionsByColor.white;
  const correctBlack = blackPreds.reduce((acc, p) => acc + (((p >= 0.5) ? 1 : 0) === blackWin ? 1 : 0), 0);
  const correctWhite = whitePreds.reduce((acc, p) => acc + (((p >= 0.5) ? 1 : 0) === whiteWin ? 1 : 0), 0);
  const sumBlack = blackPreds.reduce((acc, p) => acc + p, 0);
  const sumWhite = whitePreds.reduce((acc, p) => acc + p, 0);

  stats.games += 1;
  if (blackWin) stats.blackWins += 1; else stats.whiteWins += 1;
  stats.totalPredictions += blackPreds.length + whitePreds.length;
  stats.correctPredictions += correctBlack + correctWhite;
  stats.confidenceSum += sumBlack + sumWhite;
  stats.lastWinner = blackWin ? 'Black' : 'White';
  stats.lastScore = score;

  stats.perModel.black.predictions += blackPreds.length;
  stats.perModel.black.correctPredictions += correctBlack;
  stats.perModel.black.confidenceSum += sumBlack;
  stats.perModel.white.predictions += whitePreds.length;
  stats.perModel.white.correctPredictions += correctWhite;
  stats.perModel.white.confidenceSum += sumWhite;

  const totalPredictionsGame = blackPreds.length + whitePreds.length;
  const avgConfGame = totalPredictionsGame ? (sumBlack + sumWhite) / totalPredictionsGame : 0.5;

  const historyEntry = {
    game: stats.games,
    winRate: stats.games ? stats.blackWins / stats.games : 0,
    accuracy: stats.totalPredictions ? stats.correctPredictions / stats.totalPredictions : 0,
    avgConfidence: avgConfGame,
    score,
    blackWin: !!blackWin
  };
  gameHistory.push(historyEntry);
  if (gameHistory.length > maxHistory) {
    gameHistory = gameHistory.slice(gameHistory.length - maxHistory);
  }

  self.postMessage({
    type: 'gameComplete',
    stats: formatStats(),
    winner: stats.lastWinner,
    score,
    weights: exportWeights(),
    history: exportHistory()
  });

  currentGame = stats.games + 1;
  startNewGame();
}

function startNewGame() {
  clearTimer();
  board = new Board(config.size, config.komi);
  statesByColor = { black: [], white: [] };
  predictionsByColor = { black: [], white: [] };
  lastMove = -1;
  const confidence = evaluateBoardConfidence(board);
  self.postMessage({
    type: 'gameStart',
    board: snapshotBoard(),
    confidence,
    gameNumber: currentGame,
    running
  });
  if (running) scheduleNext();
}

function serializeNet(net) {
  if (!net) return null;
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

function exportWeights() {
  return {
    black: serializeNet(nets.black),
    white: serializeNet(nets.white)
  };
}

function exportHistory() {
  return gameHistory.map(entry => ({ ...entry }));
}

function formatStats() {
  const modelSummary = (model) => {
    const predictions = model && typeof model.predictions === 'number' ? model.predictions : 0;
    const correct = model && typeof model.correctPredictions === 'number' ? model.correctPredictions : 0;
    const confidenceSum = model && typeof model.confidenceSum === 'number' ? model.confidenceSum : 0;
    const trainingSteps = model && typeof model.trainingSteps === 'number' ? model.trainingSteps : 0;
    return {
      predictions,
      accuracy: predictions ? correct / predictions : 0,
      avgConfidence: predictions ? confidenceSum / predictions : 0,
      trainingSteps
    };
  };

  const res = {
    games: stats.games,
    blackWins: stats.blackWins,
    whiteWins: stats.whiteWins,
    blackWinRate: stats.games ? stats.blackWins / stats.games : 0,
    predictionAccuracy: stats.totalPredictions ? stats.correctPredictions / stats.totalPredictions : 0,
    totalPredictions: stats.totalPredictions,
    avgConfidence: stats.totalPredictions ? stats.confidenceSum / stats.totalPredictions : 0,
    trainingSteps: stats.trainingSteps,
    lastWinner: stats.lastWinner,
    lastScore: stats.lastScore,
    perModel: {
      black: modelSummary(stats.perModel.black),
      white: modelSummary(stats.perModel.white)
    }
  };
  return res;
}

function fullReset(messageType = 'resetDone') {
  clearTimer();
  nets = {
    black: createNet(),
    white: createNet()
  };
  stats = createStats();
  board = new Board(config.size, config.komi);
  statesByColor = { black: [], white: [] };
  predictionsByColor = { black: [], white: [] };
  lastMove = -1;
  currentGame = 1;
  gameHistory = [];
  const confidence = evaluateBoardConfidence(board);
  self.postMessage({
    type: messageType,
    board: snapshotBoard(),
    stats: formatStats(),
    weights: exportWeights(),
    confidence,
    config: { ...config },
    running,
    history: exportHistory()
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
      nets.black.setLearningRate(config.learningRate);
      nets.white.setLearningRate(config.learningRate);
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
    if (cfg.captureBias != null) {
      const bias = Math.max(0, Math.min(1, +cfg.captureBias));
      config.captureBias = bias;
    }
    if (needsReset) {
      fullReset('resetDone');
    } else {
      self.postMessage({ type: 'config', config: { ...config } });
      self.postMessage({ type: 'weights', weights: exportWeights(), history: exportHistory() });
    }
  }
};

fullReset('init');
startNewGame();
