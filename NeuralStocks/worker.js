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

const smaPeriods = [5, 10, 20];
const emaPeriods = [5, 10, 20];
const rsiPeriods = [14];
const macdConfig = { fast: 12, slow: 26, signal: 9 };

const smaSeries = smaPeriods.map(period => computeSMA(prices, period));
const emaSeries = emaPeriods.map(period => computeEMA(prices, period));
const rsiSeries = rsiPeriods.map(period => computeRSI(prices, period));
const macdSeries = computeMACDSeries(prices, macdConfig.fast, macdConfig.slow, macdConfig.signal);

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
  traderRewardScale: 120
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

let portfolio = createPortfolio();

const tradingExtraFeatureCount = 3
  + smaPeriods.length
  + emaPeriods.length
  + rsiPeriods.length
  + 3; // macd, signal, histogram

function createTraderInputSize() {
  return config.windowSize + tradingExtraFeatureCount;
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
    lastAction: 'HOLD',
    lastConfidence: 0,
    lastEdge: 0,
    exploration: config.traderExploration,
    learningRate: config.traderLearningRate
  };
}

let trader = createTrader();
let tradingStats = createTradingStats();

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
  tradingStats.exploration = config.traderExploration;
  tradingStats.learningRate = config.traderLearningRate;
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

function createPortfolioSnapshot() {
  if (!portfolio) return null;
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
    equityHistory: portfolio.equityHistory.slice()
  };
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

  applyTradingPolicy(sample, predictedPrice);

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
    })),
    portfolio: createPortfolioSnapshot(),
    trading: {
      steps: tradingStats.steps,
      avgReward: tradingStats.avgReward,
      lastReward: tradingStats.lastReward,
      lastAction: tradingStats.lastAction,
      lastConfidence: tradingStats.lastConfidence,
      lastEdge: tradingStats.lastEdge,
      exploration: tradingStats.exploration,
      learningRate: tradingStats.learningRate
    }
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
  portfolio = createPortfolio();
  trader = createTrader();
  tradingStats = createTradingStats();
  postSnapshot();
  if (resume || wasRunning) {
    setRunning(true);
  }
}

function applyConfig(newConfig) {
  const requiresReset = (
    (newConfig.hiddenUnits != null && newConfig.hiddenUnits !== config.hiddenUnits) ||
    (newConfig.windowSize != null && newConfig.windowSize !== config.windowSize) ||
    (newConfig.traderHiddenUnits != null && newConfig.traderHiddenUnits !== config.traderHiddenUnits) ||
    (newConfig.traderHiddenUnits2 != null && newConfig.traderHiddenUnits2 !== config.traderHiddenUnits2)
  );
  Object.assign(config, newConfig);
  config.windowSize = Math.max(4, Math.min(config.windowSize, Math.max(4, totalPoints - 1)));
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

  if (!requiresReset) {
    net.setLearningRate(config.learningRate);
    trader.setLearningRate(config.traderLearningRate);
    tradingStats.exploration = config.traderExploration;
    tradingStats.learningRate = config.traderLearningRate;
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
