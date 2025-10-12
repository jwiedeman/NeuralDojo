// Historical closes courtesy of Stooq. We load a multi-instrument universe captured in market_history.js
// and fall back to the legacy spy_daily.js stream if needed.
importScripts('market_history.js', 'spy_daily.js');

const earliestDateStr = '2000-01-01';
const todayStr = new Date().toISOString().slice(0, 10);

function coerceNumber(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function sanitizeSeries(entry) {
  const symbol = entry?.symbol || entry?.ticker || 'UNKNOWN';
  const name = entry?.name || symbol;
  const rawSeries = Array.isArray(entry?.series) ? entry.series : Array.isArray(entry) ? entry : [];
  const dedup = new Map();
  for (const row of rawSeries) {
    const dateStr = row?.date || row?.Date || row?.d;
    const closeVal = coerceNumber(row?.close ?? row?.Close ?? row?.c);
    if (!dateStr || !closeVal) continue;
    if (dateStr < earliestDateStr || dateStr > todayStr) continue;
    dedup.set(dateStr, closeVal);
  }
  const ordered = Array.from(dedup.entries()).sort((a, b) => a[0].localeCompare(b[0]));
  const series = ordered.map(([date, close]) => ({ date, close }));
  return { symbol, name, series };
}

const fallbackUniverse = Array.isArray(self.SPY_DAILY)
  ? [{ symbol: 'SPY', name: 'SPDR S&P 500 ETF Trust', series: self.SPY_DAILY }]
  : [];

const rawUniverse = Array.isArray(self.MARKET_HISTORY) && self.MARKET_HISTORY.length
  ? self.MARKET_HISTORY
  : fallbackUniverse;

const marketUniverse = rawUniverse
  .map(sanitizeSeries)
  .filter(entry => Array.isArray(entry.series) && entry.series.length > 260);

if (!marketUniverse.length) {
  throw new Error('No market data available');
}

let prices = new Float32Array(0);
let dates = [];
let totalPoints = 0;
let mean = 0;
let std = 1;

function computeSMA(values, period) {
  const result = new Float32Array(values.length);
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= period) {
      sum -= values[i - period];
    }
    if (i >= period - 1) {
      result[i] = sum / period;
    } else {
      result[i] = values[i];
    }
  }
  return result;
}

function computeEMA(values, period) {
  const result = new Float32Array(values.length);
  if (!values.length) return result;
  const multiplier = 2 / (period + 1);
  let ema = values[0];
  result[0] = ema;
  for (let i = 1; i < values.length; i++) {
    const value = values[i];
    ema = (value - ema) * multiplier + ema;
    result[i] = ema;
  }
  return result;
}

function computeRSI(values, period) {
  const result = new Float32Array(values.length);
  if (values.length === 0) return result;
  let gainSum = 0;
  let lossSum = 0;
  result[0] = 50;
  for (let i = 1; i < values.length; i++) {
    const change = values[i] - values[i - 1];
    const gain = Math.max(change, 0);
    const loss = Math.max(-change, 0);
    if (i < period) {
      gainSum += gain;
      lossSum += loss;
      result[i] = 50;
    } else if (i === period) {
      gainSum = (gainSum + gain) / period;
      lossSum = (lossSum + loss) / period;
    } else {
      gainSum = ((period - 1) * gainSum + gain) / period;
      lossSum = ((period - 1) * lossSum + loss) / period;
    }
    if (i >= period) {
      if (lossSum === 0) {
        result[i] = gainSum === 0 ? 50 : 100;
      } else if (gainSum === 0) {
        result[i] = 0;
      } else {
        const rs = gainSum / lossSum;
        result[i] = 100 - 100 / (1 + rs);
      }
    }
  }
  return result;
}

function computeMACDSeries(values, fastPeriod, slowPeriod, signalPeriod) {
  const fast = computeEMA(values, fastPeriod);
  const slow = computeEMA(values, slowPeriod);
  const macd = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    macd[i] = fast[i] - slow[i];
  }
  const signal = computeEMA(macd, signalPeriod);
  const histogram = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    histogram[i] = macd[i] - signal[i];
  }
  return { macd, signal, histogram };
}

const smaPeriods = [5, 10, 20, 50, 100, 200];
const emaPeriods = [5, 10, 20, 50, 100, 200];
const rsiPeriods = [6, 14, 28];
const macdConfig = { fast: 12, slow: 26, signal: 9 };
const returnPeriods = [1, 5, 10, 20];

let smaSeries = [];
let emaSeries = [];
let rsiSeries = [];
let macdSeries = { macd: new Float32Array(0), signal: new Float32Array(0), histogram: new Float32Array(0) };
let returnSeries = [];

function computeReturnSeries(values, period) {
  const result = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    if (i >= period) {
      const prev = values[i - period];
      if (prev !== 0) {
        result[i] = (values[i] - prev) / prev;
        continue;
      }
    }
    result[i] = 0;
  }
  return result;
}

function updateDatasetMoments() {
  if (!totalPoints) {
    mean = 0;
    std = 1;
    return;
  }
  let sum = 0;
  for (let i = 0; i < prices.length; i++) {
    sum += prices[i];
  }
  mean = sum / prices.length;
  let variance = 0;
  for (let i = 0; i < prices.length; i++) {
    const diff = prices[i] - mean;
    variance += diff * diff;
  }
  variance = variance / prices.length;
  std = Math.max(Math.sqrt(variance), 1e-6);
}

function refreshTechnicalSeries() {
  smaSeries = smaPeriods.map(period => computeSMA(prices, period));
  emaSeries = emaPeriods.map(period => computeEMA(prices, period));
  rsiSeries = rsiPeriods.map(period => computeRSI(prices, period));
  macdSeries = computeMACDSeries(prices, macdConfig.fast, macdConfig.slow, macdConfig.signal);
  returnSeries = returnPeriods.map(period => computeReturnSeries(prices, period));
}

function normalize(price) {
  return (price - mean) / std;
}

function denormalize(value) {
  return value * std + mean;
}

function normalizeOrZero(value) {
  return Number.isFinite(value) ? normalize(value) : 0;
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

class RLTrader {
  constructor(inputSize, hiddenUnits1, hiddenUnits2, learningRate) {
    this.inputSize = inputSize;
    this.hiddenUnits1 = hiddenUnits1;
    this.hiddenUnits2 = hiddenUnits2;
    this.actionCount = 3; // hold, buy, sell
    this.learningRate = learningRate;
    this.initWeights();
  }

  initWeights() {
    const scale1 = 1 / Math.sqrt(this.inputSize);
    const scale2 = 1 / Math.sqrt(Math.max(this.hiddenUnits1, 1));
    const scale3 = 1 / Math.sqrt(Math.max(this.hiddenUnits2, 1));
    this.w1 = new Float32Array(this.hiddenUnits1 * this.inputSize);
    this.b1 = new Float32Array(this.hiddenUnits1);
    this.w2 = new Float32Array(this.hiddenUnits2 * this.hiddenUnits1);
    this.b2 = new Float32Array(this.hiddenUnits2);
    this.w3 = new Float32Array(this.actionCount * this.hiddenUnits2);
    this.b3 = new Float32Array(this.actionCount);
    for (let i = 0; i < this.w1.length; i++) {
      this.w1[i] = (Math.random() * 2 - 1) * scale1;
    }
    for (let i = 0; i < this.w2.length; i++) {
      this.w2[i] = (Math.random() * 2 - 1) * scale2;
    }
    for (let i = 0; i < this.w3.length; i++) {
      this.w3[i] = (Math.random() * 2 - 1) * scale3;
    }
    this.b1.fill(0);
    this.b2.fill(0);
    this.b3.fill(0);
  }

  setLearningRate(lr) {
    this.learningRate = lr;
  }

  forward(features) {
    const hidden1 = new Float32Array(this.hiddenUnits1);
    for (let h = 0; h < this.hiddenUnits1; h++) {
      let sum = this.b1[h];
      const offset = h * this.inputSize;
      for (let i = 0; i < this.inputSize; i++) {
        sum += this.w1[offset + i] * features[i];
      }
      hidden1[h] = Math.tanh(sum);
    }

    const hidden2 = new Float32Array(this.hiddenUnits2);
    for (let h = 0; h < this.hiddenUnits2; h++) {
      let sum = this.b2[h];
      const offset = h * this.hiddenUnits1;
      for (let i = 0; i < this.hiddenUnits1; i++) {
        sum += this.w2[offset + i] * hidden1[i];
      }
      hidden2[h] = Math.tanh(sum);
    }

    const logits = new Float32Array(this.actionCount);
    for (let a = 0; a < this.actionCount; a++) {
      let sum = this.b3[a];
      const offset = a * this.hiddenUnits2;
      for (let h = 0; h < this.hiddenUnits2; h++) {
        sum += this.w3[offset + h] * hidden2[h];
      }
      logits[a] = sum;
    }

    const maxLogit = Math.max(...logits);
    let total = 0;
    const probs = new Float32Array(this.actionCount);
    for (let a = 0; a < this.actionCount; a++) {
      const exp = Math.exp(logits[a] - maxLogit);
      probs[a] = exp;
      total += exp;
    }
    if (total > 0) {
      for (let a = 0; a < this.actionCount; a++) {
        probs[a] /= total;
      }
    } else {
      const uniform = 1 / this.actionCount;
      probs.fill(uniform);
    }

    return { hidden1, hidden2, probs, logits };
  }

  act(features, exploration = 0) {
    const { hidden1, hidden2, probs, logits } = this.forward(features);
    let actionIndex = this.sampleFromDistribution(probs);
    if (Math.random() < exploration) {
      actionIndex = Math.floor(Math.random() * this.actionCount);
    }
    const confidence = probs[actionIndex] ?? 0;
    return {
      index: actionIndex,
      confidence,
      probs,
      logits,
      hidden1,
      hidden2,
      features
    };
  }

  sampleFromDistribution(probs) {
    const r = Math.random();
    let cume = 0;
    for (let i = 0; i < probs.length; i++) {
      cume += probs[i];
      if (r <= cume) {
        return i;
      }
    }
    return probs.length - 1;
  }

  train(decision, reward) {
    if (!decision || !decision.features || !decision.hidden1 || !decision.hidden2 || !decision.probs) return;
    const lr = this.learningRate;
    const { features, hidden1, hidden2, probs, index: actionIndex } = decision;
    const gradLogits = new Float32Array(this.actionCount);
    for (let a = 0; a < this.actionCount; a++) {
      let value = probs[a];
      if (a === actionIndex) {
        value -= 1;
      }
      gradLogits[a] = value * reward;
    }

    for (let a = 0; a < this.actionCount; a++) {
      const grad = gradLogits[a];
      this.b3[a] -= lr * grad;
      const offset = a * this.hiddenUnits2;
      for (let h = 0; h < this.hiddenUnits2; h++) {
        this.w3[offset + h] -= lr * grad * hidden2[h];
      }
    }

    const gradHidden2 = new Float32Array(this.hiddenUnits2);
    for (let h = 0; h < this.hiddenUnits2; h++) {
      let sum = 0;
      for (let a = 0; a < this.actionCount; a++) {
        const offset = a * this.hiddenUnits2 + h;
        sum += gradLogits[a] * this.w3[offset];
      }
      gradHidden2[h] = sum * (1 - hidden2[h] * hidden2[h]);
    }

    const gradHidden1 = new Float32Array(this.hiddenUnits1);
    for (let h = 0; h < this.hiddenUnits1; h++) {
      let sum = 0;
      for (let a = 0; a < this.hiddenUnits2; a++) {
        const offset = a * this.hiddenUnits1 + h;
        sum += gradHidden2[a] * this.w2[offset];
      }
      gradHidden1[h] = sum * (1 - hidden1[h] * hidden1[h]);
    }

    for (let h = 0; h < this.hiddenUnits2; h++) {
      const grad = gradHidden2[h];
      const offset = h * this.hiddenUnits1;
      for (let i = 0; i < this.hiddenUnits1; i++) {
        this.w2[offset + i] -= lr * grad * hidden1[i];
      }
      this.b2[h] -= lr * grad;
    }

    for (let h = 0; h < this.hiddenUnits1; h++) {
      const grad = gradHidden1[h];
      const offset = h * this.inputSize;
      for (let i = 0; i < this.inputSize; i++) {
        this.w1[offset + i] -= lr * grad * features[i];
      }
      this.b1[h] -= lr * grad;
    }
  }
}

const historyLimit = 240;
const recentLimit = 8;

const config = {
  learningRate: 0.05,
  hiddenUnits: 18,
  windowSize: 24,
  noise: 0.01,
  delayMs: 100,
  traderLearningRate: 0.01,
  traderHiddenUnits: 24,
  traderHiddenUnits2: 24,
  traderExploration: 0.05,
  traderRewardScale: 120,
  tickerSubsetMin: 5,
  tickerSubsetMax: 10
};

let net = null;
let running = false;
let timer = null;
let cursor = 0;
let loops = 0;
let history = {
  actual: [],
  predicted: [],
  errors: []
};
let recentPredictions = [];

const portfolioConfig = {
  initialCash: 100000,
  tradeHistoryLimit: 18,
  equityHistoryLimit: historyLimit
};

function createPortfolio() {
  return {
    initialCash: portfolioConfig.initialCash,
    cash: portfolioConfig.initialCash,
    position: 0,
    avgCost: 0,
    equity: portfolioConfig.initialCash,
    realizedPnl: 0,
    unrealizedPnl: 0,
    totalReturn: 0,
    lastPrice: null,
    trades: [],
    equityHistory: []
  };
}

let portfolio = null;

let trader = null;
let tradingStats = null;
let stats = null;
let activeDataset = null;
let playlist = [];
let activePlaylistIndex = -1;

function shuffle(array) {
  const copy = array.slice();
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function randomInt(min, max) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return min || 0;
  if (max < min) [min, max] = [max, min];
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function playlistPosition() {
  return activePlaylistIndex >= 0 ? activePlaylistIndex + 1 : 0;
}

function rebuildTickerPlaylist() {
  if (!marketUniverse.length) {
    playlist = [];
    activePlaylistIndex = -1;
    return;
  }
  const min = Math.max(1, Math.min((config.tickerSubsetMin | 0) || 1, marketUniverse.length));
  const maxCandidate = Math.max(min, Math.min((config.tickerSubsetMax | 0) || min, marketUniverse.length));
  const sampleSize = Math.max(min, Math.min(maxCandidate, randomInt(min, maxCandidate)));
  playlist = shuffle(marketUniverse).slice(0, sampleSize);
  activePlaylistIndex = -1;
}

function advanceToNextDataset(options = {}) {
  if (!playlist.length || activePlaylistIndex >= playlist.length - 1) {
    rebuildTickerPlaylist();
  }
  if (!playlist.length) {
    return false;
  }
  activePlaylistIndex += 1;
  const dataset = playlist[activePlaylistIndex];
  return setActiveDataset(dataset, options);
}

function setActiveDataset(dataset, { resetNetwork = false, resetTrader = false } = {}) {
  if (!dataset || !Array.isArray(dataset.series) || !dataset.series.length) {
    return false;
  }
  activeDataset = dataset;
  const length = dataset.series.length;
  prices = new Float32Array(length);
  dates = new Array(length);
  for (let i = 0; i < length; i++) {
    const point = dataset.series[i];
    prices[i] = Number(point.close);
    dates[i] = point.date;
  }
  totalPoints = length;
  updateDatasetMoments();
  refreshTechnicalSeries();
  const maxWindow = Math.max(4, Math.min(config.windowSize | 0 || 24, Math.max(4, totalPoints - 1)));
  if (maxWindow !== config.windowSize) {
    config.windowSize = maxWindow;
  }
  cursor = Math.max(config.windowSize, Math.min(totalPoints - 1, config.windowSize));
  if (!net || resetNetwork || net.inputSize !== config.windowSize || net.hiddenUnits !== config.hiddenUnits) {
    net = new PriceNet(config.windowSize, config.hiddenUnits, config.learningRate);
  } else {
    net.setLearningRate(config.learningRate);
  }
  const expectedTraderInput = createTraderInputSize();
  if (!trader || resetTrader || trader.inputSize !== expectedTraderInput) {
    trader = createTrader();
    tradingStats = createTradingStats();
  }
  if (!tradingStats) {
    tradingStats = createTradingStats();
  }
  trader.setLearningRate(config.traderLearningRate);
  tradingStats.exploration = config.traderExploration;
  tradingStats.learningRate = config.traderLearningRate;
  tradingStats.playlistPosition = playlistPosition();
  tradingStats.playlistSize = playlist.length;
  history = { actual: [], predicted: [], errors: [] };
  recentPredictions = [];
  portfolio = createPortfolio();
  stats = createStats();
  stats.ticker = dataset.symbol;
  stats.instrumentName = dataset.name;
  stats.playlistPosition = playlistPosition();
  stats.playlistSize = playlist.length;
  stats.availableTickers = marketUniverse.length;
  const firstDate = dataset.series[0]?.date;
  const lastDate = dataset.series[dataset.series.length - 1]?.date;
  stats.dateRange = firstDate && lastDate ? `${firstDate} → ${lastDate}` : '—';
  stats.windowCoverage = windowCoverage();
  stats.lastReset = formatClock();
  stats.loops = loops;
  return true;
}

function tradingFeatureCount() {
  return 3 + smaPeriods.length + emaPeriods.length + rsiPeriods.length + 3 + returnPeriods.length;
}

function createTraderInputSize() {
  return config.windowSize + tradingFeatureCount();
}

function createTrader() {
  return new RLTrader(
    createTraderInputSize(),
    config.traderHiddenUnits,
    config.traderHiddenUnits2,
    config.traderLearningRate
  );
}

function createTradingStats() {
  return {
    steps: 0,
    avgReward: 0,
    lastReward: 0,
    lifetimeReward: 0,
    lastAction: 'HOLD',
    lastConfidence: 0,
    lastEdge: 0,
    exploration: config.traderExploration,
    learningRate: config.traderLearningRate,
    cycleCount: 0,
    lastCycleReturn: 0,
    bestCycleReturn: 0,
    cumulativeReturn: 0,
    trades: 0,
    wins: 0,
    losses: 0,
    winRate: 0,
    playlistPosition: playlistPosition(),
    playlistSize: playlist.length
  };
}

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
    lastReset: '—',
    ticker: activeDataset?.symbol ?? '—',
    instrumentName: activeDataset?.name ?? '—',
    playlistPosition: playlistPosition(),
    playlistSize: playlist.length,
    availableTickers: marketUniverse.length,
    dateRange: activeDataset?.series?.length
      ? `${activeDataset.series[0].date} → ${activeDataset.series[activeDataset.series.length - 1].date}`
      : '—'
  };
}

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
    date: dates[index],
    index
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

function markToMarket(date, price) {
  if (!portfolio || !Number.isFinite(price)) return;
  const equity = portfolio.cash + portfolio.position * price;
  const unrealized = portfolio.position > 0 ? (price - portfolio.avgCost) * portfolio.position : 0;
  portfolio.unrealizedPnl = unrealized;
  portfolio.equity = equity;
  portfolio.totalReturn = portfolio.initialCash > 0
    ? (equity - portfolio.initialCash) / portfolio.initialCash
    : 0;
  portfolio.lastPrice = price;
  portfolio.equityHistory.push({
    date,
    equity,
    price,
    position: portfolio.position
  });
  if (portfolio.equityHistory.length > portfolioConfig.equityHistoryLimit) {
    portfolio.equityHistory.shift();
  }
}

function recordTrade({ date, side, shares, price, edgePct, pnl }) {
  if (!portfolio) return;
  const entry = {
    date,
    side,
    shares,
    price,
    edgePct,
    pnl,
    equityAfter: portfolio.equity,
    cashAfter: portfolio.cash,
    positionAfter: portfolio.position
  };
  portfolio.trades.unshift(entry);
  if (portfolio.trades.length > portfolioConfig.tradeHistoryLimit) {
    portfolio.trades.pop();
  }
  if (tradingStats) {
    tradingStats.trades = (tradingStats.trades || 0) + 1;
    if (Number.isFinite(pnl) && pnl !== 0) {
      if (pnl > 0) tradingStats.wins += 1;
      else if (pnl < 0) tradingStats.losses += 1;
    }
    const total = Math.max(1, tradingStats.trades);
    tradingStats.winRate = Math.max(0, Math.min(1, tradingStats.wins / total));
  }
}

function attemptBuy(price, fraction, date, edge) {
  if (!portfolio || !Number.isFinite(price) || price <= 0) {
    markToMarket(date, price);
    return false;
  }
  const budget = Math.min(portfolio.cash, portfolio.equity * fraction);
  const shares = Math.floor(budget / price);
  if (shares <= 0) {
    markToMarket(date, price);
    return false;
  }
  const cost = shares * price;
  const existingValue = portfolio.avgCost * portfolio.position;
  portfolio.cash -= cost;
  portfolio.position += shares;
  portfolio.avgCost = portfolio.position > 0 ? (existingValue + cost) / portfolio.position : 0;
  markToMarket(date, price);
  recordTrade({
    date,
    side: 'BUY',
    shares,
    price,
    edgePct: edge * 100,
    pnl: 0
  });
  return true;
}

function attemptSell(price, fraction, date, edge) {
  if (!portfolio || portfolio.position <= 0 || !Number.isFinite(price) || price <= 0) {
    markToMarket(date, price);
    return false;
  }
  const desiredShares = Math.max(1, Math.floor(portfolio.position * fraction));
  const shares = Math.min(desiredShares, portfolio.position);
  if (shares <= 0) {
    markToMarket(date, price);
    return false;
  }
  const proceeds = shares * price;
  portfolio.cash += proceeds;
  portfolio.position -= shares;
  const realized = portfolio.avgCost > 0 ? (price - portfolio.avgCost) * shares : 0;
  portfolio.realizedPnl += realized;
  if (portfolio.position <= 0) {
    portfolio.position = 0;
    portfolio.avgCost = 0;
  }
  markToMarket(date, price);
  recordTrade({
    date,
    side: 'SELL',
    shares,
    price,
    edgePct: edge * 100,
    pnl: realized
  });
  return true;
}

function tradingActionLabel(index) {
  if (index === 1) return 'BUY';
  if (index === 2) return 'SELL';
  return 'HOLD';
}

function buildTradingFeatures(sample, predictedPrice) {
  const features = new Float32Array(createTraderInputSize());
  features.set(sample.features);
  const predictedNorm = Number.isFinite(predictedPrice) ? normalize(predictedPrice) : 0;
  let offset = config.windowSize;
  features[offset++] = predictedNorm;
  const edge = Number.isFinite(predictedPrice) && sample.targetPrice > 0
    ? (predictedPrice - sample.targetPrice) / sample.targetPrice
    : 0;
  features[offset++] = edge;
  features[offset++] = normalizeOrZero(sample.targetPrice);
  const idx = sample.index ?? cursor;
  for (let i = 0; i < smaSeries.length; i++) {
    const value = smaSeries[i][idx];
    features[offset++] = normalizeOrZero(Number.isFinite(value) ? value : sample.targetPrice);
  }
  for (let i = 0; i < emaSeries.length; i++) {
    const value = emaSeries[i][idx];
    features[offset++] = normalizeOrZero(Number.isFinite(value) ? value : sample.targetPrice);
  }
  for (let i = 0; i < rsiSeries.length; i++) {
    const value = rsiSeries[i][idx];
    const normalizedRsi = Number.isFinite(value) ? (value - 50) / 50 : 0;
    features[offset++] = Math.max(-2, Math.min(2, normalizedRsi));
  }
  const macdVal = macdSeries.macd[idx];
  const signalVal = macdSeries.signal[idx];
  const histVal = macdSeries.histogram[idx];
  features[offset++] = Number.isFinite(macdVal) ? macdVal / std : 0;
  features[offset++] = Number.isFinite(signalVal) ? signalVal / std : 0;
  features[offset++] = Number.isFinite(histVal) ? histVal / std : 0;
  for (let i = 0; i < returnSeries.length; i++) {
    const value = returnSeries[i][idx];
    const clamped = Number.isFinite(value) ? Math.max(-3, Math.min(3, value)) : 0;
    features[offset++] = clamped;
  }
  return { features, edge };
}

function updateTradingStats(decision, reward, edge) {
  tradingStats.steps += 1;
  tradingStats.lastAction = tradingActionLabel(decision?.index ?? 0);
  const confidence = decision?.confidence ?? 0;
  tradingStats.lastConfidence = Math.max(0, Math.min(1, confidence));
  tradingStats.lastEdge = Number.isFinite(edge) ? edge : 0;
  const safeReward = Number.isFinite(reward) ? reward : 0;
  tradingStats.lastReward = safeReward;
  const step = tradingStats.steps;
  tradingStats.avgReward += (safeReward - tradingStats.avgReward) / step;
  tradingStats.lifetimeReward += safeReward;
  tradingStats.exploration = config.traderExploration;
  tradingStats.learningRate = config.traderLearningRate;
  tradingStats.playlistPosition = playlistPosition();
  tradingStats.playlistSize = playlist.length;
}

function applyTradingPolicy(sample, predictedPrice) {
  if (!portfolio || !sample || !Number.isFinite(sample.targetPrice)) return;
  const { features, edge } = buildTradingFeatures(sample, predictedPrice);
  const decision = trader.act(features, config.traderExploration);
  const price = sample.targetPrice;
  const prevEquity = portfolio.equity;
  let traded = false;
  const minFraction = 0.05;
  const maxFraction = 0.5;
  const fraction = Math.min(maxFraction, minFraction + (maxFraction - minFraction) * Math.max(0, decision.confidence));
  if (decision.index === 1) {
    traded = attemptBuy(price, fraction, sample.date, edge);
  } else if (decision.index === 2) {
    traded = attemptSell(price, fraction, sample.date, edge);
  }
  if (!traded) {
    markToMarket(sample.date, price);
  }
  const equityChange = portfolio.equity - prevEquity;
  const normalizedReward = equityChange / Math.max(1, portfolio.initialCash);
  const scaledReward = Math.max(-5, Math.min(5, normalizedReward * config.traderRewardScale));
  trader.train(decision, scaledReward);
  updateTradingStats(decision, normalizedReward, edge);
}

function computeEquityMetrics(history) {
  if (!Array.isArray(history) || history.length === 0) {
    return { maxDrawdown: 0, sharpe: 0 };
  }
  let peak = null;
  let maxDrawdown = 0;
  const returns = [];
  for (let i = 0; i < history.length; i++) {
    const equity = Number(history[i]?.equity);
    if (!Number.isFinite(equity)) continue;
    if (peak == null || equity > peak) {
      peak = equity;
    }
    if (peak > 0) {
      const drawdown = (equity - peak) / peak;
      if (drawdown < maxDrawdown) {
        maxDrawdown = drawdown;
      }
    }
    if (i > 0) {
      const prev = Number(history[i - 1]?.equity);
      if (Number.isFinite(prev) && prev > 0) {
        returns.push((equity - prev) / prev);
      }
    }
  }
  let sharpe = 0;
  if (returns.length > 1) {
    const meanRet = returns.reduce((sum, v) => sum + v, 0) / returns.length;
    const variance = returns.reduce((sum, v) => sum + Math.pow(v - meanRet, 2), 0) / (returns.length - 1);
    const volatility = Math.sqrt(Math.max(variance, 0));
    if (volatility > 0) {
      sharpe = (meanRet / volatility) * Math.sqrt(252);
    }
  }
  return { maxDrawdown, sharpe };
}

function createPortfolioSnapshot() {
  if (!portfolio) return null;
  const metrics = computeEquityMetrics(portfolio.equityHistory);
  return {
    equity: portfolio.equity,
    cash: portfolio.cash,
    position: portfolio.position,
    avgCost: portfolio.avgCost,
    unrealizedPnl: portfolio.unrealizedPnl,
    realizedPnl: portfolio.realizedPnl,
    totalReturn: portfolio.totalReturn,
    lastPrice: portfolio.lastPrice,
    trades: portfolio.trades.map(trade => ({ ...trade })),
    equityHistory: portfolio.equityHistory.slice(),
    maxDrawdown: metrics.maxDrawdown,
    sharpe: metrics.sharpe,
    tradeCount: tradingStats?.trades ?? portfolio.trades.length,
    winRate: tradingStats?.winRate ?? 0
  };
}

function completeCycle() {
  if (!tradingStats) {
    tradingStats = createTradingStats();
  }
  if (portfolio) {
    const finalReturn = Number.isFinite(portfolio.totalReturn) ? portfolio.totalReturn : 0;
    tradingStats.cycleCount += 1;
    tradingStats.lastCycleReturn = finalReturn;
    tradingStats.cumulativeReturn += finalReturn;
    if (tradingStats.cycleCount === 1) {
      tradingStats.bestCycleReturn = finalReturn;
    } else {
      tradingStats.bestCycleReturn = Math.max(tradingStats.bestCycleReturn, finalReturn);
    }
  }
  loops += 1;
  if (stats) {
    stats.loops = loops;
  }
}

function stepOnce() {
  if ((!activeDataset || !totalPoints) && !advanceToNextDataset({ resetNetwork: true, resetTrader: true })) {
    setRunning(false);
    return;
  }
  if (totalPoints <= config.windowSize) {
    setRunning(false);
    return;
  }
  if (cursor >= totalPoints) {
    completeCycle();
    if (!advanceToNextDataset({ resetNetwork: false, resetTrader: false })) {
      setRunning(false);
      return;
    }
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

  applyTradingPolicy(sample, predictedPrice);

  cursor += 1;
  const progressDenom = Math.max(1, totalPoints - config.windowSize);
  const relativeCursor = Math.max(0, cursor - config.windowSize);
  stats.progressPct = progressDenom <= 0 ? 0 : Math.min(100, (relativeCursor / progressDenom) * 100);

  postSnapshot();
}

function postSnapshot() {
  if (!stats) {
    stats = createStats();
  }
  stats.ticker = activeDataset?.symbol ?? stats.ticker ?? '—';
  stats.instrumentName = activeDataset?.name ?? stats.instrumentName ?? '—';
  stats.playlistPosition = playlistPosition();
  stats.playlistSize = playlist.length;
  stats.availableTickers = marketUniverse.length;
  stats.windowCoverage = windowCoverage();

  const rmse = Math.sqrt(Math.max(stats.mse ?? 0, 0));
  const bestMae = stats.bestMae === Infinity ? null : stats.bestMae;

  const weightsSnapshot = net
    ? {
        inputWeights: Array.from(net.w1),
        outputWeights: Array.from(net.w2),
        hiddenUnits: net.hiddenUnits,
        inputSize: net.inputSize
      }
    : {
        inputWeights: [],
        outputWeights: [],
        hiddenUnits: config.hiddenUnits,
        inputSize: config.windowSize
      };

  if (!tradingStats) {
    tradingStats = createTradingStats();
  }
  tradingStats.playlistPosition = playlistPosition();
  tradingStats.playlistSize = playlist.length;

  const tradingSummary = {
    steps: tradingStats.steps,
    avgReward: tradingStats.avgReward,
    lastReward: tradingStats.lastReward,
    lastAction: tradingStats.lastAction,
    lastConfidence: tradingStats.lastConfidence,
    lastEdge: tradingStats.lastEdge,
    exploration: tradingStats.exploration,
    learningRate: tradingStats.learningRate,
    cycleCount: tradingStats.cycleCount,
    lastCycleReturn: tradingStats.lastCycleReturn,
    bestCycleReturn: tradingStats.bestCycleReturn,
    lifetimeReward: tradingStats.lifetimeReward,
    cumulativeReturn: tradingStats.cumulativeReturn,
    trades: tradingStats.trades,
    winRate: tradingStats.winRate,
    playlistPosition: tradingStats.playlistPosition,
    playlistSize: tradingStats.playlistSize
  };

  const snapshot = {
    history: {
      actual: history.actual.slice(),
      predicted: history.predicted.slice(),
      errors: history.errors.slice()
    },
    stats: {
      ...stats,
      rmse,
      bestMae
    },
    weights: weightsSnapshot,
    recentPredictions: recentPredictions.map(item => ({
      label: item.label,
      actual: item.actual,
      predicted: item.predicted,
      error: item.error
    })),
    portfolio: createPortfolioSnapshot(),
    trading: tradingSummary
  };
  self.postMessage({ type: 'snapshot', snapshot });
}

function resetState(resume = false) {
  const wasRunning = running;
  if (wasRunning) setRunning(false);
  net = null;
  history = { actual: [], predicted: [], errors: [] };
  recentPredictions = [];
  loops = 0;
  cursor = 0;
  stats = null;
  portfolio = null;
  trader = null;
  tradingStats = null;
  activeDataset = null;
  rebuildTickerPlaylist();
  const initialized = advanceToNextDataset({ resetNetwork: true, resetTrader: true });
  if (!initialized) {
    stats = createStats();
  }
  if (stats) {
    stats.lastReset = formatClock();
  }
  postSnapshot();
  if (resume || wasRunning) {
    setRunning(true);
  }
}

function applyConfig(newConfig) {
  const prevHiddenUnits = config.hiddenUnits;
  const prevWindowSize = config.windowSize;
  const prevTraderHU1 = config.traderHiddenUnits;
  const prevTraderHU2 = config.traderHiddenUnits2;
  const prevTickerSubsetMin = config.tickerSubsetMin;
  const prevTickerSubsetMax = config.tickerSubsetMax;

  Object.assign(config, newConfig);

  config.windowSize = Math.max(4, Math.min(config.windowSize, Math.max(4, totalPoints - 1 || config.windowSize)));
  config.hiddenUnits = Math.max(2, config.hiddenUnits | 0);
  config.learningRate = Math.max(1e-4, config.learningRate);
  config.noise = Math.max(0, config.noise);
  config.delayMs = Math.max(0, config.delayMs | 0);
  config.traderLearningRate = Math.max(1e-4, Number.isFinite(config.traderLearningRate) ? config.traderLearningRate : 0.01);
  config.traderHiddenUnits = Math.max(2, Number.isFinite(config.traderHiddenUnits) ? (config.traderHiddenUnits | 0) : 24);
  config.traderHiddenUnits2 = Math.max(2, Number.isFinite(config.traderHiddenUnits2) ? (config.traderHiddenUnits2 | 0) : 24);
  config.traderExploration = Math.min(1, Math.max(0, Number.isFinite(config.traderExploration) ? config.traderExploration : 0.05));
  config.traderRewardScale = Number.isFinite(config.traderRewardScale)
    ? Math.max(1, config.traderRewardScale)
    : 120;
  config.tickerSubsetMin = Math.max(1, Math.min((config.tickerSubsetMin | 0) || 1, marketUniverse.length));
  config.tickerSubsetMax = Math.max(config.tickerSubsetMin, Math.min((config.tickerSubsetMax | 0) || config.tickerSubsetMin, marketUniverse.length));

  const requiresReset = (
    config.hiddenUnits !== prevHiddenUnits ||
    config.windowSize !== prevWindowSize ||
    config.traderHiddenUnits !== prevTraderHU1 ||
    config.traderHiddenUnits2 !== prevTraderHU2 ||
    config.tickerSubsetMin !== prevTickerSubsetMin ||
    config.tickerSubsetMax !== prevTickerSubsetMax
  );

  if (!requiresReset) {
    if (net) {
      net.setLearningRate(config.learningRate);
    }
    if (trader) {
      trader.setLearningRate(config.traderLearningRate);
    }
    if (!tradingStats) {
      tradingStats = createTradingStats();
    }
    tradingStats.exploration = config.traderExploration;
    tradingStats.learningRate = config.traderLearningRate;
    if (stats) {
      stats.learningRate = config.learningRate;
      stats.noise = config.noise;
      stats.windowSize = config.windowSize;
      stats.hiddenUnits = config.hiddenUnits;
      stats.windowCoverage = windowCoverage();
    }
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

// Prime the environment so the UI can render immediately.
resetState(false);
