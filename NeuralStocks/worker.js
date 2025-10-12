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
const logReturnPeriods = [1, 5, 20];
const realizedVolPeriods = [10, 20];
const atrPeriods = [14, 21];
const skewPeriods = [20];
const kurtosisPeriods = [20];
const volatilityZLookback = 60;
const calendarDowCount = 5;
const calendarMonthCount = 12;

let smaSeries = [];
let emaSeries = [];
let rsiSeries = [];
let macdSeries = { macd: new Float32Array(0), signal: new Float32Array(0), histogram: new Float32Array(0) };
let returnSeries = [];
let logReturnSeries = [];
let realizedVolSeries = [];
let downsideVolSeries = [];
let atrSeries = [];
let skewSeries = [];
let kurtosisSeries = [];
let distanceToSma200Series = new Float32Array(0);
let gapSeries = new Float32Array(0);
let trendRegimeSeries = new Float32Array(0);
let volatilityZScoreSeries = new Float32Array(0);
let volatilityRegimeSeries = new Float32Array(0);
let dayOfWeekOneHotSeries = [];
let monthOneHotSeries = [];

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

function computeLogReturnSeries(values, period) {
  const result = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    if (i >= period && values[i - period] > 0 && values[i] > 0) {
      result[i] = Math.log(values[i] / values[i - period]);
    } else {
      result[i] = 0;
    }
  }
  return result;
}

function computeRealizedVolatility(returns, period) {
  const result = new Float32Array(returns.length);
  let sumSq = 0;
  for (let i = 0; i < returns.length; i++) {
    const value = Number.isFinite(returns[i]) ? returns[i] : 0;
    sumSq += value * value;
    if (i >= period) {
      const prev = Number.isFinite(returns[i - period]) ? returns[i - period] : 0;
      sumSq -= prev * prev;
    }
    const window = Math.min(i + 1, period);
    const variance = window > 0 ? Math.max(sumSq / window, 0) : 0;
    result[i] = Math.sqrt(variance) * Math.sqrt(252);
  }
  return result;
}

function computeDownsideVolatility(returns, period) {
  const result = new Float32Array(returns.length);
  for (let i = 0; i < returns.length; i++) {
    const start = Math.max(0, i - period + 1);
    let sumSq = 0;
    let count = 0;
    for (let j = start; j <= i; j++) {
      const value = Number.isFinite(returns[j]) ? returns[j] : 0;
      if (value < 0) {
        sumSq += value * value;
        count += 1;
      }
    }
    if (count > 0) {
      result[i] = Math.sqrt(sumSq / count) * Math.sqrt(252);
    } else {
      result[i] = 0;
    }
  }
  return result;
}

function computeRollingAtr(values, period) {
  const result = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const start = Math.max(1, i - period + 1);
    let sum = 0;
    let count = 0;
    for (let j = start; j <= i; j++) {
      const prev = values[j - 1];
      const curr = values[j];
      if (Number.isFinite(prev) && Number.isFinite(curr)) {
        sum += Math.abs(curr - prev);
        count += 1;
      }
    }
    result[i] = count > 0 ? sum / count : 0;
  }
  return result;
}

function computeRollingSkew(values, period) {
  const result = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const start = Math.max(0, i - period + 1);
    const window = [];
    for (let j = start; j <= i; j++) {
      const value = Number.isFinite(values[j]) ? values[j] : 0;
      window.push(value);
    }
    if (window.length < 3) {
      result[i] = 0;
      continue;
    }
    const mean = window.reduce((sum, v) => sum + v, 0) / window.length;
    let variance = 0;
    let thirdMoment = 0;
    for (const value of window) {
      const diff = value - mean;
      variance += diff * diff;
      thirdMoment += diff * diff * diff;
    }
    variance = variance / window.length;
    const std = Math.sqrt(Math.max(variance, 0));
    if (std === 0) {
      result[i] = 0;
      continue;
    }
    result[i] = (thirdMoment / window.length) / Math.pow(std, 3);
  }
  return result;
}

function computeRollingKurtosis(values, period) {
  const result = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const start = Math.max(0, i - period + 1);
    const window = [];
    for (let j = start; j <= i; j++) {
      const value = Number.isFinite(values[j]) ? values[j] : 0;
      window.push(value);
    }
    if (window.length < 4) {
      result[i] = 0;
      continue;
    }
    const mean = window.reduce((sum, v) => sum + v, 0) / window.length;
    let variance = 0;
    let fourthMoment = 0;
    for (const value of window) {
      const diff = value - mean;
      const diffSq = diff * diff;
      variance += diffSq;
      fourthMoment += diffSq * diffSq;
    }
    variance = variance / window.length;
    const std = Math.sqrt(Math.max(variance, 0));
    if (std === 0) {
      result[i] = 0;
      continue;
    }
    const excess = fourthMoment / window.length / Math.pow(std, 4) - 3;
    result[i] = Number.isFinite(excess) ? excess : 0;
  }
  return result;
}

function computeDistanceToSeries(values, referenceSeries) {
  const result = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const price = values[i];
    const ref = referenceSeries?.[i];
    if (Number.isFinite(price) && Number.isFinite(ref) && ref !== 0) {
      result[i] = (price - ref) / ref;
    } else {
      result[i] = 0;
    }
  }
  return result;
}

function computeGapSeries(values) {
  const result = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    if (i === 0) {
      result[i] = 0;
      continue;
    }
    const prev = values[i - 1];
    const curr = values[i];
    if (Number.isFinite(prev) && Number.isFinite(curr) && prev !== 0) {
      result[i] = (curr - prev) / prev;
    } else {
      result[i] = 0;
    }
  }
  return result;
}

function computeRollingZScore(values, period) {
  const result = new Float32Array(values.length);
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < values.length; i++) {
    const value = Number.isFinite(values[i]) ? values[i] : 0;
    sum += value;
    sumSq += value * value;
    if (i >= period) {
      const prev = Number.isFinite(values[i - period]) ? values[i - period] : 0;
      sum -= prev;
      sumSq -= prev * prev;
    }
    const count = Math.min(i + 1, period);
    const mean = count > 0 ? sum / count : 0;
    const variance = count > 0 ? Math.max(sumSq / count - mean * mean, 0) : 0;
    const std = Math.sqrt(variance);
    result[i] = std > 0 ? (value - mean) / std : 0;
  }
  return result;
}

function computeTrendRegime(values, slowSeries, fastSeries) {
  const result = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const price = values[i];
    const slow = slowSeries?.[i];
    const fast = fastSeries?.[i];
    if (Number.isFinite(price) && Number.isFinite(slow)) {
      const aboveSlow = price >= slow ? 1 : -1;
      if (Number.isFinite(fast)) {
        const fastAboveSlow = fast >= slow ? 1 : -1;
        result[i] = (aboveSlow + fastAboveSlow) / 2;
      } else {
        result[i] = aboveSlow;
      }
    } else {
      result[i] = 0;
    }
  }
  return result;
}

function computeVolatilityRegime(zScores) {
  const result = new Float32Array(zScores.length);
  for (let i = 0; i < zScores.length; i++) {
    const z = Number.isFinite(zScores[i]) ? zScores[i] : 0;
    if (z > 0.75) result[i] = 1;
    else if (z < -0.75) result[i] = -1;
    else result[i] = 0;
  }
  return result;
}

function parseDateSafe(dateStr) {
  if (!dateStr) return null;
  const parsed = new Date(`${dateStr}T00:00:00Z`);
  return Number.isFinite(parsed?.getTime?.()) ? parsed : null;
}

function daysBetween(startStr, endStr) {
  const start = parseDateSafe(startStr);
  const end = parseDateSafe(endStr);
  if (!start || !end) return 0;
  const diffMs = end.getTime() - start.getTime();
  return Math.max(0, Math.round(diffMs / (1000 * 60 * 60 * 24)));
}

function computeCalendarOneHot(dateArray, count, mode) {
  const series = Array.from({ length: count }, () => new Float32Array(dateArray.length));
  for (let i = 0; i < dateArray.length; i++) {
    const date = parseDateSafe(dateArray[i]);
    if (!date) continue;
    if (mode === 'dow') {
      const day = date.getUTCDay();
      if (day >= 1 && day <= 5) {
        series[day - 1][i] = 1;
      }
    } else if (mode === 'month') {
      const month = date.getUTCMonth();
      if (month >= 0 && month < count) {
        series[month][i] = 1;
      }
    }
  }
  return series;
}

function estimateTransactionCost(shares, price, idx, edge = 0) {
  if (!Number.isFinite(price) || price <= 0 || !Number.isFinite(shares) || shares <= 0) {
    return 0;
  }
  const notional = shares * price;
  const vol10 = realizedVolSeries[0]?.[idx] ?? 0;
  const vol20 = realizedVolSeries[realizedVolSeries.length - 1]?.[idx] ?? vol10;
  const vol = Math.max(0, Number.isFinite(vol10) ? vol10 : 0, Number.isFinite(vol20) ? vol20 : 0);
  const volImpact = Math.min(config.slippageVolCap, vol * config.slippageVolCoeff);
  const zScore = Number.isFinite(volatilityZScoreSeries[idx]) ? Math.abs(volatilityZScoreSeries[idx]) : 0;
  const zImpact = Math.min(config.slippageVolCap, zScore * config.slippageZCoeff);
  const edgeImpact = Math.min(config.slippageVolCap, Math.abs(edge) * config.edgeImpactCoeff);
  const variableCost = notional * (config.costPerSideBps + volImpact + zImpact + edgeImpact);
  const totalCost = Math.max(config.minCommission, variableCost);
  return Math.min(notional * 0.1, totalCost);
}

function updateHoldDurations(sharesSold, sellIndex) {
  if (!portfolio || !Array.isArray(portfolio.openLots) || sharesSold <= 0) {
    return null;
  }
  let remaining = sharesSold;
  let weightedDays = 0;
  let sharesAccounted = 0;
  while (remaining > 0 && portfolio.openLots.length > 0) {
    const lot = portfolio.openLots[0];
    if (!lot) break;
    const consume = Math.min(remaining, lot.shares);
    const holdDays = daysBetween(dates[lot.index], dates[sellIndex]);
    weightedDays += holdDays * consume;
    sharesAccounted += consume;
    portfolio.totalHoldDays += holdDays * consume;
    portfolio.closedPositions += consume;
    lot.shares -= consume;
    if (lot.shares <= 0) {
      portfolio.openLots.shift();
    }
    remaining -= consume;
  }
  portfolio.avgHoldDays = portfolio.closedPositions > 0 ? portfolio.totalHoldDays / portfolio.closedPositions : 0;
  if (sharesAccounted > 0) {
    return weightedDays / sharesAccounted;
  }
  return null;
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
  logReturnSeries = logReturnPeriods.map(period => computeLogReturnSeries(prices, period));
  const oneDayReturns = computeReturnSeries(prices, 1);
  realizedVolSeries = realizedVolPeriods.map(period => computeRealizedVolatility(oneDayReturns, period));
  downsideVolSeries = realizedVolPeriods.map(period => computeDownsideVolatility(oneDayReturns, period));
  atrSeries = atrPeriods.map(period => computeRollingAtr(prices, period));
  skewSeries = skewPeriods.map(period => computeRollingSkew(oneDayReturns, period));
  kurtosisSeries = kurtosisPeriods.map(period => computeRollingKurtosis(oneDayReturns, period));
  const slowIdx = smaPeriods.indexOf(200);
  const slowSeries = slowIdx >= 0 ? smaSeries[slowIdx] : null;
  distanceToSma200Series = computeDistanceToSeries(prices, slowSeries);
  gapSeries = computeGapSeries(prices);
  const fastIdx = smaPeriods.indexOf(50);
  const fastSeries = fastIdx >= 0 ? smaSeries[fastIdx] : null;
  trendRegimeSeries = computeTrendRegime(prices, slowSeries, fastSeries);
  const volReference = realizedVolSeries.length ? realizedVolSeries[realizedVolSeries.length - 1] : new Float32Array(prices.length);
  volatilityZScoreSeries = computeRollingZScore(volReference, volatilityZLookback);
  volatilityRegimeSeries = computeVolatilityRegime(volatilityZScoreSeries);
  dayOfWeekOneHotSeries = computeCalendarOneHot(dates, calendarDowCount, 'dow');
  monthOneHotSeries = computeCalendarOneHot(dates, calendarMonthCount, 'month');
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
  traderBaseFraction: 0.05,
  traderMaxFraction: 0.5,
  traderMinTradeFraction: 0.01,
  traderTargetVol: 0.25,
  traderMaxExposure: 0.75,
  traderMinEdge: 0.002,
  traderTrendThreshold: 0.1,
  traderMaxVolZ: 2.5,
  traderMaxRealizedVol: 0.6,
  traderCooldownBars: 3,
  traderStopoutReturn: -0.02,
  tickerSubsetMin: 5,
  tickerSubsetMax: 10,
  costPerSideBps: 0.0005,
  minCommission: 0,
  slippageVolCoeff: 0.02,
  slippageVolCap: 0.003,
  slippageZCoeff: 0.0005,
  edgeImpactCoeff: 0.0005
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

const riskState = {
  cooldown: 0,
  lastStopReturn: null,
  lastStopReason: null
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
    equityHistory: [],
    costsPaid: 0,
    totalValueTraded: 0,
    totalVolume: 0,
    tradeCountTotal: 0,
    closedTradeCount: 0,
    hitCount: 0,
    winPnls: [],
    lossPnls: [],
    openLots: [],
    totalHoldDays: 0,
    closedPositions: 0,
    avgHoldDays: 0
  };
}

function resetRiskState() {
  riskState.cooldown = 0;
  riskState.lastStopReturn = null;
  riskState.lastStopReason = null;
}

function computePortfolioExposure(price) {
  if (!portfolio || !Number.isFinite(price) || price <= 0) {
    return 0;
  }
  const equity = Number.isFinite(portfolio.equity) && portfolio.equity > 0 ? portfolio.equity : portfolio.cash;
  if (!Number.isFinite(equity) || equity <= 0) {
    return 0;
  }
  return Math.max(0, (portfolio.position * price) / equity);
}

function computeTradeFraction(confidence, gating, price, action) {
  const minFraction = Math.max(0, Math.min(1, config.traderBaseFraction));
  const maxFraction = Math.max(minFraction, Math.min(1, config.traderMaxFraction));
  const minTrade = Math.max(0, Math.min(1, config.traderMinTradeFraction));
  const boundedConfidence = Math.max(0, Math.min(1, confidence ?? 0));
  const range = Math.max(0, maxFraction - minFraction);
  let fraction = minFraction + range * boundedConfidence;
  if (action === 1) {
    const volatility = Number.isFinite(gating?.volatility) ? Math.max(gating.volatility, 1e-6) : null;
    if (volatility && config.traderTargetVol > 0) {
      const volScale = Math.min(1, config.traderTargetVol / volatility);
      fraction *= volScale;
    }
    fraction = Math.min(fraction, maxFraction);
    const remainingExposure = Math.max(0, config.traderMaxExposure - (gating?.exposure ?? 0));
    fraction = Math.min(fraction, remainingExposure);
    if (fraction < minTrade) {
      return remainingExposure >= minTrade ? minTrade : 0;
    }
    return Math.max(0, fraction);
  }
  if (fraction < minTrade) {
    fraction = minTrade;
  }
  return Math.max(0, Math.min(1, fraction));
}

function registerStopout(returnPct, reason = 'stop') {
  if (!Number.isFinite(returnPct)) {
    return;
  }
  if (returnPct <= config.traderStopoutReturn) {
    const cooldown = Math.max(0, config.traderCooldownBars | 0);
    if (cooldown > 0) {
      riskState.cooldown = Math.max(riskState.cooldown, cooldown);
      riskState.lastStopReturn = returnPct;
      riskState.lastStopReason = reason;
    }
  }
}

function evaluatePositionGating(sample, edge, desiredAction) {
  const idx = sample.index ?? cursor;
  const price = sample.targetPrice;
  const trend = Number.isFinite(trendRegimeSeries[idx]) ? trendRegimeSeries[idx] : 0;
  const volBucket = Number.isFinite(volatilityRegimeSeries[idx]) ? volatilityRegimeSeries[idx] : 0;
  const volZ = Number.isFinite(volatilityZScoreSeries[idx]) ? volatilityZScoreSeries[idx] : 0;
  const vol10 = realizedVolSeries[0]?.[idx];
  const vol20 = realizedVolSeries[realizedVolSeries.length - 1]?.[idx];
  const realizedVol = Math.max(
    0,
    Number.isFinite(vol10) ? vol10 : 0,
    Number.isFinite(vol20) ? vol20 : 0
  );
  const exposure = computePortfolioExposure(price);

  const gating = {
    desiredAction,
    action: desiredAction,
    reasons: [],
    trend,
    volBucket,
    volZ,
    volatility: realizedVol,
    exposure,
    cooldown: riskState.cooldown,
    edge: edge ?? 0,
    minEdge: config.traderMinEdge,
    trendThreshold: config.traderTrendThreshold,
    volatilityCap: config.traderMaxRealizedVol,
    volZCap: config.traderMaxVolZ,
    targetVol: config.traderTargetVol,
    maxExposure: config.traderMaxExposure,
    cooldownReason: riskState.lastStopReason,
    lastStopReturn: riskState.lastStopReturn
  };

  const absEdge = Math.abs(edge ?? 0);
  const minEdge = config.traderMinEdge;
  const trendThreshold = config.traderTrendThreshold;
  const maxVolZ = config.traderMaxVolZ;
  const maxVol = config.traderMaxRealizedVol;

  if (desiredAction === 1) {
    if (riskState.cooldown > 0) gating.reasons.push('cooldown');
    if (absEdge < minEdge) gating.reasons.push('edge');
    if (trend < trendThreshold) gating.reasons.push('trend');
    if (Math.abs(volZ) > maxVolZ) gating.reasons.push('vol_z');
    if (realizedVol > maxVol) gating.reasons.push('volatility');
    if (exposure >= config.traderMaxExposure - 1e-6) gating.reasons.push('exposure');
    if (gating.reasons.length) {
      gating.action = 0;
    }
  } else if (desiredAction === 2) {
    if (!portfolio || portfolio.position <= 0) {
      gating.reasons.push('flat');
      gating.action = 0;
    } else {
      if (edge <= -minEdge) gating.reasons.push('edge');
      if (trend <= -trendThreshold) gating.reasons.push('trend');
      if (Math.abs(volZ) > maxVolZ) gating.reasons.push('vol_z');
      if (realizedVol > maxVol) gating.reasons.push('volatility');
      if (!gating.reasons.length) {
        gating.reasons.push('hold');
        gating.action = 0;
      }
    }
  } else {
    gating.action = 0;
  }

  gating.allowed = gating.action === desiredAction;
  gating.primaryReason = gating.reasons[0] ?? null;
  return gating;
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
  resetRiskState();
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
  return (
    3 +
    smaPeriods.length +
    emaPeriods.length +
    rsiPeriods.length +
    3 +
    returnPeriods.length +
    logReturnPeriods.length +
    realizedVolPeriods.length +
    realizedVolPeriods.length +
    skewPeriods.length +
    kurtosisPeriods.length +
    atrPeriods.length +
    1 + // distance to SMA200
    1 + // gap size
    1 + // trend regime
    1 + // volatility z-score
    1 + // volatility regime bucket
    calendarDowCount +
    calendarMonthCount
  );
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
    playlistSize: playlist.length,
    regime: 0,
    volBucket: 0,
    volZ: 0,
    realizedVol: 0,
    minEdge: config.traderMinEdge,
    trendThreshold: config.traderTrendThreshold,
    volatilityCap: config.traderMaxRealizedVol,
    volZCap: config.traderMaxVolZ,
    targetVol: config.traderTargetVol,
    maxExposure: config.traderMaxExposure,
    cooldown: riskState.cooldown,
    cooldownReason: riskState.lastStopReason ?? null,
    gateReason: null
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

function recordTrade({ date, side, shares, price, edgePct, pnl, costs, holdingDays }) {
  if (!portfolio) return;
  const entry = {
    date,
    side,
    shares,
    price,
    edgePct,
    pnl,
    costs,
    holdingDays,
    equityAfter: portfolio.equity,
    cashAfter: portfolio.cash,
    positionAfter: portfolio.position
  };
  portfolio.trades.unshift(entry);
  if (portfolio.trades.length > portfolioConfig.tradeHistoryLimit) {
    portfolio.trades.pop();
  }
  portfolio.tradeCountTotal = (portfolio.tradeCountTotal || 0) + 1;
  if (side === 'SELL' && Number.isFinite(pnl)) {
    portfolio.closedTradeCount = (portfolio.closedTradeCount || 0) + 1;
    if (pnl > 0) {
      portfolio.hitCount = (portfolio.hitCount || 0) + 1;
      portfolio.winPnls.push(pnl);
      if (portfolio.winPnls.length > portfolioConfig.tradeHistoryLimit) {
        portfolio.winPnls.shift();
      }
    } else if (pnl < 0) {
      portfolio.lossPnls.push(pnl);
      if (portfolio.lossPnls.length > portfolioConfig.tradeHistoryLimit) {
        portfolio.lossPnls.shift();
      }
    }
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

function attemptBuy(price, fraction, date, edge, idx) {
  if (!portfolio || !Number.isFinite(price) || price <= 0) {
    markToMarket(date, price);
    return false;
  }
  const budget = Math.min(portfolio.cash, portfolio.equity * fraction);
  let shares = Math.floor(budget / price);
  while (shares > 0) {
    const cost = shares * price;
    const tradeCost = estimateTransactionCost(shares, price, idx, edge);
    if (cost + tradeCost <= portfolio.cash + 1e-6) {
      break;
    }
    shares -= 1;
  }
  if (shares <= 0) {
    markToMarket(date, price);
    return false;
  }
  const cost = shares * price;
  const tradeCost = estimateTransactionCost(shares, price, idx, edge);
  const totalCost = cost + tradeCost;
  if (totalCost > portfolio.cash) {
    markToMarket(date, price);
    return false;
  }
  const existingValue = portfolio.avgCost * portfolio.position;
  portfolio.cash -= totalCost;
  portfolio.position += shares;
  portfolio.avgCost = portfolio.position > 0 ? (existingValue + cost) / portfolio.position : 0;
  portfolio.costsPaid += tradeCost;
  portfolio.totalValueTraded += cost;
  portfolio.totalVolume += shares;
  portfolio.openLots.push({ shares, index: idx });
  markToMarket(date, price);
  recordTrade({
    date,
    side: 'BUY',
    shares,
    price,
    edgePct: edge * 100,
    pnl: 0,
    costs: tradeCost
  });
  return true;
}

function attemptSell(price, fraction, date, edge, idx) {
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
  const tradeCost = estimateTransactionCost(shares, price, idx, edge);
  const netProceeds = proceeds - tradeCost;
  const prevAvgCost = portfolio.avgCost;
  portfolio.cash += netProceeds;
  portfolio.position -= shares;
  const realizedGross = prevAvgCost > 0 ? (price - prevAvgCost) * shares : 0;
  const realized = realizedGross - tradeCost;
  portfolio.realizedPnl += realized;
  if (portfolio.position <= 0) {
    portfolio.position = 0;
    portfolio.avgCost = 0;
  }
  portfolio.costsPaid += tradeCost;
  portfolio.totalValueTraded += proceeds;
  portfolio.totalVolume += shares;
  const holdingDays = updateHoldDurations(shares, idx);
  markToMarket(date, price);
  const costBasis = prevAvgCost > 0 ? prevAvgCost * shares : 0;
  const returnPct = costBasis > 0 ? realized / costBasis : 0;
  registerStopout(returnPct, realized < 0 ? 'drawdown' : 'trim');
  recordTrade({
    date,
    side: 'SELL',
    shares,
    price,
    edgePct: edge * 100,
    pnl: realized,
    costs: tradeCost,
    holdingDays
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
  for (let i = 0; i < logReturnSeries.length; i++) {
    const value = logReturnSeries[i][idx];
    const clamped = Number.isFinite(value) ? Math.max(-3, Math.min(3, value)) : 0;
    features[offset++] = clamped;
  }
  for (let i = 0; i < realizedVolSeries.length; i++) {
    const value = realizedVolSeries[i][idx];
    features[offset++] = Number.isFinite(value) ? Math.min(5, Math.max(0, value)) : 0;
  }
  for (let i = 0; i < downsideVolSeries.length; i++) {
    const value = downsideVolSeries[i][idx];
    features[offset++] = Number.isFinite(value) ? Math.min(5, Math.max(0, value)) : 0;
  }
  for (let i = 0; i < skewSeries.length; i++) {
    const value = skewSeries[i][idx];
    features[offset++] = Number.isFinite(value) ? Math.max(-5, Math.min(5, value)) : 0;
  }
  for (let i = 0; i < kurtosisSeries.length; i++) {
    const value = kurtosisSeries[i][idx];
    features[offset++] = Number.isFinite(value) ? Math.max(-5, Math.min(5, value)) : 0;
  }
  for (let i = 0; i < atrSeries.length; i++) {
    const value = atrSeries[i][idx];
    const normalized = Number.isFinite(value) && sample.targetPrice > 0 ? value / sample.targetPrice : 0;
    features[offset++] = Math.max(0, Math.min(5, normalized));
  }
  const distance = distanceToSma200Series[idx];
  features[offset++] = Number.isFinite(distance) ? Math.max(-5, Math.min(5, distance)) : 0;
  const gap = gapSeries[idx];
  features[offset++] = Number.isFinite(gap) ? Math.max(-5, Math.min(5, gap)) : 0;
  const trendRegime = trendRegimeSeries[idx];
  features[offset++] = Number.isFinite(trendRegime) ? Math.max(-1, Math.min(1, trendRegime)) : 0;
  const volZ = volatilityZScoreSeries[idx];
  features[offset++] = Number.isFinite(volZ) ? Math.max(-5, Math.min(5, volZ)) : 0;
  const volBucket = volatilityRegimeSeries[idx];
  features[offset++] = Number.isFinite(volBucket) ? Math.max(-1, Math.min(1, volBucket)) : 0;
  for (let i = 0; i < dayOfWeekOneHotSeries.length; i++) {
    features[offset++] = dayOfWeekOneHotSeries[i]?.[idx] ?? 0;
  }
  for (let i = 0; i < monthOneHotSeries.length; i++) {
    features[offset++] = monthOneHotSeries[i]?.[idx] ?? 0;
  }
  return { features, edge };
}

function updateTradingStats(decision, reward, edge, gating) {
  if (!tradingStats) {
    tradingStats = createTradingStats();
  }
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
  tradingStats.minEdge = config.traderMinEdge;
  tradingStats.trendThreshold = config.traderTrendThreshold;
  tradingStats.volatilityCap = config.traderMaxRealizedVol;
  tradingStats.volZCap = config.traderMaxVolZ;
  tradingStats.targetVol = config.traderTargetVol;
  tradingStats.maxExposure = config.traderMaxExposure;
  tradingStats.cooldown = Math.max(0, riskState.cooldown);
  tradingStats.cooldownReason = riskState.lastStopReason ?? null;
  if (gating) {
    if (Number.isFinite(gating.trend)) {
      tradingStats.regime = gating.trend;
    }
    if (Number.isFinite(gating.volBucket)) {
      tradingStats.volBucket = gating.volBucket;
    }
    if (Number.isFinite(gating.volZ)) {
      tradingStats.volZ = gating.volZ;
    }
    if (Number.isFinite(gating.volatility)) {
      tradingStats.realizedVol = gating.volatility;
    }
    if (Array.isArray(gating.reasons) && gating.reasons.length) {
      tradingStats.gateReason = gating.reasons[0];
    } else if (gating.primaryReason) {
      tradingStats.gateReason = gating.primaryReason;
    } else {
      tradingStats.gateReason = null;
    }
  } else {
    tradingStats.gateReason = null;
  }
}

function applyTradingPolicy(sample, predictedPrice) {
  if (!portfolio || !sample || !Number.isFinite(sample.targetPrice)) return;
  const idx = sample.index ?? cursor;
  const { features, edge } = buildTradingFeatures(sample, predictedPrice);
  const rawDecision = trader.act(features, config.traderExploration);
  const gating = evaluatePositionGating(sample, edge, rawDecision.index);
  const executedDecision = {
    ...rawDecision,
    originalIndex: rawDecision.index,
    index: gating.action
  };
  const price = sample.targetPrice;
  const prevEquity = portfolio.equity;
  let traded = false;
  let fraction = 0;
  if (executedDecision.index === 1) {
    fraction = computeTradeFraction(executedDecision.confidence, gating, price, 1);
    if (fraction > 0) {
      traded = attemptBuy(price, fraction, sample.date, edge, idx);
    }
    if (!traded) {
      gating.reasons.push('sizing');
      executedDecision.index = 0;
    }
  } else if (executedDecision.index === 2) {
    fraction = computeTradeFraction(executedDecision.confidence, gating, price, 2);
    if (fraction > 0) {
      traded = attemptSell(price, fraction, sample.date, edge, idx);
    }
    if (!traded) {
      gating.reasons.push('sizing');
      executedDecision.index = 0;
    }
  }
  if (!traded) {
    markToMarket(sample.date, price);
  }
  const equityChange = portfolio.equity - prevEquity;
  const normalizedReward = equityChange / Math.max(1, portfolio.initialCash);
  const scaledReward = Math.max(-5, Math.min(5, normalizedReward * config.traderRewardScale));
  trader.train(executedDecision, scaledReward);
  gating.traded = traded;
  gating.fraction = fraction;
  gating.executedAction = executedDecision.index;
  gating.originalAction = executedDecision.originalIndex;
  updateTradingStats(executedDecision, normalizedReward, edge, gating);
  if (riskState.cooldown > 0 && riskState.cooldown <= (gating.cooldown ?? riskState.cooldown)) {
    riskState.cooldown = Math.max(0, riskState.cooldown - 1);
    if (riskState.cooldown === 0) {
      riskState.lastStopReason = null;
      riskState.lastStopReturn = null;
    }
  }
}

function computeEquityMetrics(history) {
  if (!Array.isArray(history) || history.length === 0) {
    return {
      maxDrawdown: 0,
      sharpe: 0,
      sortino: 0,
      calmar: 0,
      cagr: 0,
      downsideDeviation: 0
    };
  }
  let peak = null;
  let maxDrawdown = 0;
  const returns = [];
  const downsideReturns = [];
  let startEquity = null;
  let endEquity = null;
  let startDate = null;
  let endDate = null;
  for (let i = 0; i < history.length; i++) {
    const entry = history[i];
    const equity = Number(entry?.equity);
    if (!Number.isFinite(equity)) continue;
    if (startEquity == null) {
      startEquity = equity;
      startDate = entry?.date ?? null;
    }
    endEquity = equity;
    endDate = entry?.date ?? endDate;
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
        const ret = (equity - prev) / prev;
        returns.push(ret);
        if (ret < 0) {
          downsideReturns.push(ret);
        }
      }
    }
  }
  let sharpe = 0;
  let sortino = 0;
  let downsideDeviation = 0;
  if (returns.length > 1) {
    const meanRet = returns.reduce((sum, v) => sum + v, 0) / returns.length;
    const variance = returns.reduce((sum, v) => sum + Math.pow(v - meanRet, 2), 0) / (returns.length - 1);
    const volatility = Math.sqrt(Math.max(variance, 0));
    if (volatility > 0) {
      sharpe = (meanRet / volatility) * Math.sqrt(252);
    }
    if (downsideReturns.length > 0) {
      const downsideVar = downsideReturns.reduce((sum, v) => sum + v * v, 0) / downsideReturns.length;
      downsideDeviation = Math.sqrt(Math.max(downsideVar, 0)) * Math.sqrt(252);
      if (downsideDeviation > 0) {
        sortino = (meanRet * 252) / downsideDeviation;
      }
    }
  }
  const start = Number.isFinite(startEquity) ? startEquity : null;
  const end = Number.isFinite(endEquity) ? endEquity : null;
  let cagr = 0;
  if (start && end && start > 0 && end > 0) {
    const diffDays = Math.max(1, daysBetween(startDate, endDate));
    const years = diffDays / 365.25;
    if (years > 0) {
      cagr = Math.pow(end / start, 1 / years) - 1;
    }
  }
  const calmar = maxDrawdown < 0 ? cagr / Math.abs(maxDrawdown) : 0;
  return { maxDrawdown, sharpe, sortino, calmar, cagr, downsideDeviation };
}

function createPortfolioSnapshot() {
  if (!portfolio) return null;
  const metrics = computeEquityMetrics(portfolio.equityHistory);
  const tradeCount = portfolio.tradeCountTotal || portfolio.trades.length;
  const closedTrades = portfolio.closedTradeCount || 0;
  const hitRate = closedTrades > 0 ? (portfolio.hitCount || 0) / closedTrades : 0;
  const avgWin = portfolio.winPnls.length
    ? portfolio.winPnls.reduce((sum, v) => sum + v, 0) / portfolio.winPnls.length
    : null;
  const avgLoss = portfolio.lossPnls.length
    ? portfolio.lossPnls.reduce((sum, v) => sum + v, 0) / portfolio.lossPnls.length
    : null;
  const winLossRatio = Number.isFinite(avgWin) && Number.isFinite(avgLoss) && avgLoss !== 0
    ? avgWin / Math.abs(avgLoss)
    : null;
  const turnover = portfolio.initialCash > 0 ? portfolio.totalValueTraded / portfolio.initialCash : 0;
  const costDrag = portfolio.totalValueTraded > 0 ? portfolio.costsPaid / portfolio.totalValueTraded : 0;
  const capacityPerShare = portfolio.totalVolume > 0 ? portfolio.costsPaid / portfolio.totalVolume : 0;
  const avgHoldDays = portfolio.avgHoldDays ?? 0;
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
    sortino: metrics.sortino,
    calmar: metrics.calmar,
    cagr: metrics.cagr,
    downsideDeviation: metrics.downsideDeviation,
    tradeCount,
    winRate: hitRate,
    hitRate,
    closedTradeCount: closedTrades,
    avgWin,
    avgLoss,
    winLossRatio,
    turnover,
    costDrag,
    capacityPerShare,
    avgHoldDays,
    costsPaid: portfolio.costsPaid
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
    playlistSize: tradingStats.playlistSize,
    regime: tradingStats.regime,
    volBucket: tradingStats.volBucket,
    volZ: tradingStats.volZ,
    realizedVol: tradingStats.realizedVol,
    minEdge: tradingStats.minEdge,
    trendThreshold: tradingStats.trendThreshold,
    volatilityCap: tradingStats.volatilityCap,
    volZCap: tradingStats.volZCap,
    targetVol: tradingStats.targetVol,
    maxExposure: tradingStats.maxExposure,
    cooldown: tradingStats.cooldown,
    cooldownReason: tradingStats.cooldownReason,
    gateReason: tradingStats.gateReason
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
  resetRiskState();
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
  config.traderBaseFraction = Number.isFinite(config.traderBaseFraction)
    ? Math.max(0, Math.min(1, config.traderBaseFraction))
    : 0.05;
  config.traderMaxFraction = Number.isFinite(config.traderMaxFraction)
    ? Math.max(config.traderBaseFraction, Math.min(1, config.traderMaxFraction))
    : Math.max(config.traderBaseFraction, 0.5);
  config.traderMinTradeFraction = Number.isFinite(config.traderMinTradeFraction)
    ? Math.max(0, Math.min(1, config.traderMinTradeFraction))
    : 0.01;
  config.traderTargetVol = Number.isFinite(config.traderTargetVol)
    ? Math.max(0, config.traderTargetVol)
    : 0.25;
  config.traderMaxExposure = Number.isFinite(config.traderMaxExposure)
    ? Math.max(config.traderBaseFraction, Math.min(1.5, config.traderMaxExposure))
    : 0.75;
  config.traderMinEdge = Number.isFinite(config.traderMinEdge)
    ? Math.max(0, config.traderMinEdge)
    : 0.002;
  config.traderTrendThreshold = Number.isFinite(config.traderTrendThreshold)
    ? Math.max(-1, Math.min(1, config.traderTrendThreshold))
    : 0.1;
  config.traderMaxVolZ = Number.isFinite(config.traderMaxVolZ)
    ? Math.max(0, config.traderMaxVolZ)
    : 2.5;
  config.traderMaxRealizedVol = Number.isFinite(config.traderMaxRealizedVol)
    ? Math.max(0, config.traderMaxRealizedVol)
    : 0.6;
  config.traderCooldownBars = Number.isFinite(config.traderCooldownBars)
    ? Math.max(0, Math.round(config.traderCooldownBars))
    : 3;
  config.traderStopoutReturn = Number.isFinite(config.traderStopoutReturn)
    ? Math.min(0, config.traderStopoutReturn)
    : -0.02;
  config.tickerSubsetMin = Math.max(1, Math.min((config.tickerSubsetMin | 0) || 1, marketUniverse.length));
  config.tickerSubsetMax = Math.max(config.tickerSubsetMin, Math.min((config.tickerSubsetMax | 0) || config.tickerSubsetMin, marketUniverse.length));
  config.costPerSideBps = Number.isFinite(config.costPerSideBps) ? Math.max(0, config.costPerSideBps) : 0.0005;
  config.minCommission = Number.isFinite(config.minCommission) ? Math.max(0, config.minCommission) : 0;
  config.slippageVolCoeff = Number.isFinite(config.slippageVolCoeff) ? Math.max(0, config.slippageVolCoeff) : 0.02;
  config.slippageVolCap = Number.isFinite(config.slippageVolCap) ? Math.max(0, config.slippageVolCap) : 0.003;
  config.slippageZCoeff = Number.isFinite(config.slippageZCoeff) ? Math.max(0, config.slippageZCoeff) : 0.0005;
  config.edgeImpactCoeff = Number.isFinite(config.edgeImpactCoeff) ? Math.max(0, config.edgeImpactCoeff) : 0.0005;

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
    tradingStats.minEdge = config.traderMinEdge;
    tradingStats.trendThreshold = config.traderTrendThreshold;
    tradingStats.volatilityCap = config.traderMaxRealizedVol;
    tradingStats.volZCap = config.traderMaxVolZ;
    tradingStats.targetVol = config.traderTargetVol;
    tradingStats.maxExposure = config.traderMaxExposure;
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
