// SPY daily closes courtesy of Stooq (360 most recent sessions captured into spy_daily.js).
importScripts('spy_daily.js');

const rawData = Array.isArray(self.SPY_DAILY) ? self.SPY_DAILY : [];
const dataset = rawData.map(entry => ({
  date: entry.date,
  close: Number(entry.close)
})).filter(entry => Number.isFinite(entry.close));

const prices = dataset.map(d => d.close);
const dates = dataset.map(d => d.date);
const totalPoints = prices.length;

const mean = totalPoints ? prices.reduce((sum, v) => sum + v, 0) / totalPoints : 0;
const variance = totalPoints
  ? prices.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / totalPoints
  : 0;
const std = Math.max(Math.sqrt(variance), 1e-6);

function normalize(price) {
  return (price - mean) / std;
}

function denormalize(value) {
  return value * std + mean;
}

class PriceNet {
  constructor(inputSize, hiddenUnits, learningRate) {
    this.inputSize = inputSize;
    this.hiddenUnits = hiddenUnits;
    this.learningRate = learningRate;
    this.initWeights();
  }

  initWeights() {
    const scale1 = 1 / Math.sqrt(this.inputSize);
    const scale2 = 1 / Math.sqrt(this.hiddenUnits);
    this.w1 = new Float32Array(this.hiddenUnits * this.inputSize);
    this.b1 = new Float32Array(this.hiddenUnits);
    this.w2 = new Float32Array(this.hiddenUnits);
    this.b2 = 0;
    for (let i = 0; i < this.w1.length; i++) {
      this.w1[i] = (Math.random() * 2 - 1) * scale1;
    }
    for (let i = 0; i < this.w2.length; i++) {
      this.w2[i] = (Math.random() * 2 - 1) * scale2;
    }
    this.b1.fill(0);
    this.b2 = 0;
  }

  setLearningRate(lr) {
    this.learningRate = lr;
  }

  forward(features) {
    const hidden = new Float32Array(this.hiddenUnits);
    for (let h = 0; h < this.hiddenUnits; h++) {
      let sum = this.b1[h];
      const offset = h * this.inputSize;
      for (let i = 0; i < this.inputSize; i++) {
        sum += this.w1[offset + i] * features[i];
      }
      hidden[h] = Math.tanh(sum);
    }
    let output = this.b2;
    for (let h = 0; h < this.hiddenUnits; h++) {
      output += this.w2[h] * hidden[h];
    }
    return { hidden, output };
  }

  trainSample(features, target) {
    const { hidden, output } = this.forward(features);
    const error = output - target;
    const lr = this.learningRate;

    const gradHidden = new Float32Array(this.hiddenUnits);
    for (let h = 0; h < this.hiddenUnits; h++) {
      gradHidden[h] = error * this.w2[h] * (1 - hidden[h] * hidden[h]);
    }

    for (let h = 0; h < this.hiddenUnits; h++) {
      const delta = error * hidden[h];
      this.w2[h] -= lr * delta;
    }
    this.b2 -= lr * error;

    for (let h = 0; h < this.hiddenUnits; h++) {
      const offset = h * this.inputSize;
      const grad = gradHidden[h];
      for (let i = 0; i < this.inputSize; i++) {
        this.w1[offset + i] -= lr * grad * features[i];
      }
      this.b1[h] -= lr * grad;
    }

    return { error, output };
  }
}

const historyLimit = 240;
const recentLimit = 8;

const config = {
  learningRate: 0.05,
  hiddenUnits: 18,
  windowSize: 24,
  noise: 0.01,
  delayMs: 100
};

let net = new PriceNet(config.windowSize, config.hiddenUnits, config.learningRate);
let running = false;
let timer = null;
let cursor = Math.min(totalPoints, config.windowSize);
let loops = 0;
let history = {
  actual: [],
  predicted: [],
  errors: []
};
let recentPredictions = [];

function formatClock() {
  const now = new Date();
  return now.toLocaleTimeString([], { hour12: false });
}

function createStats() {
  return {
    steps: 0,
    mae: 0,
    mse: 0,
    bestMae: Infinity,
    lastActual: null,
    lastPredicted: null,
    lastAbsError: null,
    pointsSeen: 0,
    loops: 0,
    learningRate: config.learningRate,
    noise: config.noise,
    windowSize: config.windowSize,
    hiddenUnits: config.hiddenUnits,
    progressPct: 0,
    windowCoverage: windowCoverage(),
    lastReset: formatClock()
  };
}

let stats = createStats();

function windowCoverage() {
  if (!totalPoints) return '—';
  const pct = (config.windowSize / totalPoints) * 100;
  return `${config.windowSize} / ${totalPoints} (${pct.toFixed(1)}%)`;
}

function scheduleNext() {
  if (!running) return;
  const delay = Math.max(0, config.delayMs);
  timer = setTimeout(() => {
    timer = null;
    stepOnce();
    if (running) scheduleNext();
  }, delay);
}

function setRunning(value) {
  const desired = value && totalPoints > config.windowSize ? true : false;
  if (running === desired) {
    if (value !== desired) {
      self.postMessage({ type: 'status', running: running });
    }
    return;
  }
  running = desired;
  if (!running && timer != null) {
    clearTimeout(timer);
    timer = null;
  }
  if (running) {
    scheduleNext();
  }
  self.postMessage({ type: 'status', running });
}

function sampleAt(index) {
  const features = new Float32Array(config.windowSize);
  const start = index - config.windowSize;
  for (let i = 0; i < config.windowSize; i++) {
    let value = normalize(prices[start + i]);
    if (config.noise > 0) {
      value += (Math.random() * 2 - 1) * config.noise;
    }
    features[i] = value;
  }
  const targetPrice = prices[index];
  return {
    features,
    targetNorm: normalize(targetPrice),
    targetPrice,
    date: dates[index]
  };
}

function pushHistory(actual, predicted, absError) {
  history.actual.push(actual);
  history.predicted.push(predicted);
  history.errors.push(absError);
  if (history.actual.length > historyLimit) {
    history.actual.shift();
    history.predicted.shift();
    history.errors.shift();
  }
}

function pushRecent(entry) {
  recentPredictions.unshift(entry);
  if (recentPredictions.length > recentLimit) {
    recentPredictions.pop();
  }
}

function stepOnce() {
  if (totalPoints <= config.windowSize) {
    setRunning(false);
    return;
  }
  if (cursor >= totalPoints) {
    cursor = config.windowSize;
    loops += 1;
    stats.loops = loops;
  }
  const sample = sampleAt(cursor);
  const { error, output } = net.trainSample(sample.features, sample.targetNorm);
  const predictedPrice = denormalize(output);
  const diff = predictedPrice - sample.targetPrice;
  const absError = Math.abs(diff);

  stats.steps += 1;
  stats.pointsSeen += 1;
  stats.mae += (absError - stats.mae) / stats.steps;
  const sqError = diff * diff;
  stats.mse += (sqError - stats.mse) / stats.steps;
  stats.bestMae = Math.min(stats.bestMae, stats.mae);
  stats.lastActual = sample.targetPrice;
  stats.lastPredicted = predictedPrice;
  stats.lastAbsError = absError;
  stats.learningRate = config.learningRate;
  stats.noise = config.noise;
  stats.windowSize = config.windowSize;
  stats.hiddenUnits = config.hiddenUnits;
  stats.windowCoverage = windowCoverage();

  pushHistory(sample.targetPrice, predictedPrice, absError);
  pushRecent({
    label: sample.date,
    actual: sample.targetPrice,
    predicted: predictedPrice,
    error: diff
  });

  cursor += 1;
  if (cursor >= totalPoints) {
    cursor = config.windowSize;
    loops += 1;
    stats.loops = loops;
  }

  const progressDenom = Math.max(1, totalPoints - config.windowSize);
  const relativeCursor = Math.max(0, cursor - config.windowSize);
  stats.progressPct = progressDenom <= 0 ? 0 : (relativeCursor / progressDenom) * 100;

  postSnapshot();
}

function postSnapshot() {
  const snapshot = {
    history: {
      actual: history.actual.slice(),
      predicted: history.predicted.slice(),
      errors: history.errors.slice()
    },
    stats: {
      ...stats,
      rmse: Math.sqrt(Math.max(stats.mse, 0)),
      bestMae: stats.bestMae === Infinity ? null : stats.bestMae
    },
    weights: {
      inputWeights: Array.from(net.w1),
      outputWeights: Array.from(net.w2),
      hiddenUnits: net.hiddenUnits,
      inputSize: net.inputSize
    },
    recentPredictions: recentPredictions.map(item => ({
      label: item.label,
      actual: item.actual,
      predicted: item.predicted,
      error: item.error
    }))
  };
  self.postMessage({ type: 'snapshot', snapshot });
}

function resetState(resume = false) {
  const wasRunning = running;
  if (wasRunning) setRunning(false);
  net = new PriceNet(config.windowSize, config.hiddenUnits, config.learningRate);
  history = { actual: [], predicted: [], errors: [] };
  recentPredictions = [];
  loops = 0;
  cursor = Math.min(totalPoints, config.windowSize);
  stats = createStats();
  stats.lastReset = formatClock();
  postSnapshot();
  if (resume || wasRunning) {
    setRunning(true);
  }
}

function applyConfig(newConfig) {
  const requiresReset = (
    (newConfig.hiddenUnits != null && newConfig.hiddenUnits !== config.hiddenUnits) ||
    (newConfig.windowSize != null && newConfig.windowSize !== config.windowSize)
  );
  Object.assign(config, newConfig);
  config.windowSize = Math.max(4, Math.min(config.windowSize, Math.max(4, totalPoints - 1)));
  config.hiddenUnits = Math.max(2, config.hiddenUnits | 0);
  config.learningRate = Math.max(1e-4, config.learningRate);
  config.noise = Math.max(0, config.noise);
  config.delayMs = Math.max(0, config.delayMs | 0);

  if (!requiresReset) {
    net.setLearningRate(config.learningRate);
    stats.learningRate = config.learningRate;
    stats.noise = config.noise;
    stats.windowSize = config.windowSize;
    stats.hiddenUnits = config.hiddenUnits;
    stats.windowCoverage = windowCoverage();
    postSnapshot();
  } else {
    resetState();
  }
}

self.addEventListener('message', ev => {
  const msg = ev.data || {};
  if (msg.type === 'start') {
    setRunning(true);
  } else if (msg.type === 'pause') {
    setRunning(false);
  } else if (msg.type === 'reset') {
    resetState(false);
  } else if (msg.type === 'config' && msg.config) {
    applyConfig(msg.config);
  }
});

// Emit an initial snapshot so the UI can render immediately.
postSnapshot();
