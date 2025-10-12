import { DashboardView } from './js/dashboard-view.js';

const UI = window.NeuralUI || {};

const elements = {
  priceCanvas: document.getElementById('priceChart'),
  errorCanvas: document.getElementById('errorChart'),
  weightsCanvas: document.getElementById('weightsCanvas'),
  outputCanvas: document.getElementById('outputCanvas'),
  traderInputCanvas: document.getElementById('traderInputCanvas'),
  traderHiddenCanvas: document.getElementById('traderHiddenCanvas'),
  traderOutputCanvas: document.getElementById('traderOutputCanvas'),
  latestActualEl: document.getElementById('latestActual'),
  latestPredEl: document.getElementById('latestPred'),
  latestErrorEl: document.getElementById('latestError'),
  datasetProgressEl: document.getElementById('datasetProgress'),
  windowSizeLabelEl: document.getElementById('windowSizeLabel'),
  hiddenUnitsLabelEl: document.getElementById('hiddenUnitsLabel'),
  stepCountEl: document.getElementById('stepCount'),
  maeEl: document.getElementById('mae'),
  rmseEl: document.getElementById('rmse'),
  bestMaeEl: document.getElementById('bestMae'),
  lrLabelEl: document.getElementById('lrLabel'),
  noiseLabelEl: document.getElementById('noiseLabel'),
  pointsStreamedEl: document.getElementById('pointsStreamed'),
  windowCoverageEl: document.getElementById('windowCoverage'),
  loopCountEl: document.getElementById('loopCount'),
  bestMaeSecondaryEl: document.getElementById('bestMaeSecondary'),
  lastResetEl: document.getElementById('lastReset'),
  instrumentLabelEl: document.getElementById('instrumentLabel'),
  datasetDateRangeEl: document.getElementById('datasetDateRange'),
  datasetPlaylistEl: document.getElementById('datasetPlaylist'),
  datasetUniverseEl: document.getElementById('datasetUniverse'),
  recentListEl: document.getElementById('recentList'),
  weightsRawEl: document.getElementById('weightsRaw'),
  portfolioEquityEl: document.getElementById('portfolioEquity'),
  portfolioCashEl: document.getElementById('portfolioCash'),
  portfolioPositionEl: document.getElementById('portfolioPosition'),
  portfolioAvgCostEl: document.getElementById('portfolioAvgCost'),
  portfolioUnrealizedEl: document.getElementById('portfolioUnrealized'),
  portfolioReturnEl: document.getElementById('portfolioReturn'),
  portfolioRealizedEl: document.getElementById('portfolioRealized'),
  portfolioDrawdownEl: document.getElementById('portfolioDrawdown'),
  portfolioSharpeEl: document.getElementById('portfolioSharpe'),
  portfolioSortinoEl: document.getElementById('portfolioSortino'),
  portfolioCalmarEl: document.getElementById('portfolioCalmar'),
  portfolioCagrEl: document.getElementById('portfolioCagr'),
  portfolioDownsideEl: document.getElementById('portfolioDownside'),
  portfolioTurnoverEl: document.getElementById('portfolioTurnover'),
  portfolioAvgWinLossEl: document.getElementById('portfolioAvgWinLoss'),
  portfolioAvgHoldEl: document.getElementById('portfolioAvgHold'),
  portfolioCostDragEl: document.getElementById('portfolioCostDrag'),
  portfolioCapacityEl: document.getElementById('portfolioCapacity'),
  portfolioTradeCountEl: document.getElementById('portfolioTradeCount'),
  portfolioWinRateEl: document.getElementById('portfolioWinRate'),
  portfolioCostsPaidEl: document.getElementById('portfolioCostsPaid'),
  tradeLogEl: document.getElementById('tradeLog'),
  tradingLastActionEl: document.getElementById('tradingLastAction'),
  tradingConfidenceEl: document.getElementById('tradingConfidence'),
  tradingEdgeEl: document.getElementById('tradingEdge'),
  tradingLastRewardEl: document.getElementById('tradingLastReward'),
  tradingAvgRewardEl: document.getElementById('tradingAvgReward'),
  tradingExplorationEl: document.getElementById('tradingExploration'),
  tradingCycleCountEl: document.getElementById('tradingCycleCount'),
  tradingLastCycleEl: document.getElementById('tradingLastCycle'),
  tradingBestCycleEl: document.getElementById('tradingBestCycle'),
  tradingLifetimeRewardEl: document.getElementById('tradingLifetimeReward'),
  tradingCumulativeReturnEl: document.getElementById('tradingCumulativeReturn'),
  tradingTradeCountEl: document.getElementById('tradingTradeCount'),
  tradingWinRateEl: document.getElementById('tradingWinRate'),
  tradingRegimeEl: document.getElementById('tradingRegime'),
  tradingVolatilityEl: document.getElementById('tradingVolatility'),
  tradingRiskEl: document.getElementById('tradingRisk'),
  tradingThresholdEl: document.getElementById('tradingThreshold'),
  forecastHorizonLabelEl: document.getElementById('forecastHorizonLabel'),
  forwardTableBodyEl: document.getElementById('forwardTableBody'),
  forwardEmptyEl: document.getElementById('forwardEmpty')
};

const view = new DashboardView(elements);

const startBtn = document.getElementById('startBtn');
const pauseBtn = document.getElementById('pauseBtn');
const resetBtn = document.getElementById('resetBtn');

const learningRateEl = document.getElementById('learningRate');
const learningRateValEl = document.getElementById('learningRateVal');
const hiddenUnitsEl = document.getElementById('hiddenUnits');
const hiddenUnitsValEl = document.getElementById('hiddenUnitsVal');
const windowSizeEl = document.getElementById('windowSize');
const windowSizeValEl = document.getElementById('windowSizeVal');
const noiseEl = document.getElementById('noise');
const noiseValEl = document.getElementById('noiseVal');
const delayEl = document.getElementById('delayMs');
const delayValEl = document.getElementById('delayMsVal');

const headerEl = document.querySelector('header');
const fatalErrorEl = document.createElement('div');
fatalErrorEl.className = 'fatal-error';
fatalErrorEl.style.display = 'none';
fatalErrorEl.setAttribute('role', 'alert');
if (headerEl) {
  headerEl.insertAdjacentElement('afterend', fatalErrorEl);
} else {
  document.body.prepend(fatalErrorEl);
}

const workerUrl = new URL('./worker.js', import.meta.url).toString();

const config = {
  learningRate: parseFloat(learningRateEl?.value ?? '0.05'),
  hiddenUnits: parseInt(hiddenUnitsEl?.value ?? '18', 10),
  windowSize: parseInt(windowSizeEl?.value ?? '24', 10),
  noise: parseFloat(noiseEl?.value ?? '0.01'),
  delayMs: parseInt(delayEl?.value ?? '100', 10)
};

let running = false;
let worker = null;
let fatalError = false;

function showFatalError(message, error) {
  if (error) {
    console.error(message, error);
  } else {
    console.error(message);
  }
  fatalErrorEl.textContent = message;
  fatalErrorEl.style.display = 'block';
  fatalError = true;
  if (startBtn) startBtn.disabled = true;
  if (pauseBtn) pauseBtn.disabled = true;
  if (resetBtn) resetBtn.disabled = true;
}

function setRunningState(value) {
  running = value;
  if (fatalError) {
    if (startBtn) startBtn.disabled = true;
    if (pauseBtn) pauseBtn.disabled = true;
    return;
  }
  if (startBtn) startBtn.disabled = running;
  if (pauseBtn) pauseBtn.disabled = !running;
}

function initWorker() {
  if (typeof Worker !== 'function') {
    showFatalError('This browser does not support Web Workers, which are required for this experience.');
    return null;
  }
  try {
    const instance = new Worker(workerUrl);
    instance.onmessage = ev => {
      const msg = ev.data;
      if (!msg) return;
      if (msg.type === 'status') {
        setRunningState(!!msg.running);
      } else if (msg.type === 'snapshot') {
        view.applySnapshot(msg.snapshot);
      }
    };
    instance.onerror = event => {
      showFatalError('Background worker crashed — check the console for details.', event?.error || event?.message || event);
      worker = null;
    };
    instance.onmessageerror = event => {
      showFatalError('Received malformed data from the background worker.', event?.data ?? event);
      worker = null;
    };
    return instance;
  } catch (error) {
    try {
      const fallbackInstance = new Worker(workerUrl, { type: 'classic' });
      fallbackInstance.onmessage = ev => {
        const msg = ev.data;
        if (!msg) return;
        if (msg.type === 'status') {
          setRunningState(!!msg.running);
        } else if (msg.type === 'snapshot') {
          view.applySnapshot(msg.snapshot);
        }
      };
      fallbackInstance.onerror = event => {
        showFatalError('Background worker crashed — check the console for details.', event?.error || event?.message || event);
        worker = null;
      };
      fallbackInstance.onmessageerror = event => {
        showFatalError('Received malformed data from the background worker.', event?.data ?? event);
        worker = null;
      };
      return fallbackInstance;
    } catch (fallbackError) {
      showFatalError('Unable to start the background worker. This experience cannot run.', fallbackError);
      return null;
    }
  }
}

worker = initWorker();

function postConfig(update) {
  Object.assign(config, update);
  if (worker) {
    worker.postMessage({ type: 'config', config: { ...config } });
  }
}

startBtn?.addEventListener('click', () => {
  if (worker) {
    worker.postMessage({ type: 'start' });
  }
});

pauseBtn?.addEventListener('click', () => {
  if (worker) {
    worker.postMessage({ type: 'pause' });
  }
});

resetBtn?.addEventListener('click', () => {
  if (worker) {
    worker.postMessage({ type: 'reset' });
  }
});

UI.bindRangeControl?.(learningRateEl, {
  valueEl: learningRateValEl,
  format: v => v.toFixed(3),
  onCommit: v => postConfig({ learningRate: v })
});

UI.bindRangeControl?.(hiddenUnitsEl, {
  valueEl: hiddenUnitsValEl,
  format: v => Math.round(v),
  onCommit: v => postConfig({ hiddenUnits: Math.round(v) })
});

UI.bindRangeControl?.(windowSizeEl, {
  valueEl: windowSizeValEl,
  format: v => Math.round(v),
  onCommit: v => postConfig({ windowSize: Math.round(v) })
});

UI.bindRangeControl?.(noiseEl, {
  valueEl: noiseValEl,
  format: v => v.toFixed(3),
  onCommit: v => postConfig({ noise: v })
});

UI.bindRangeControl?.(delayEl, {
  valueEl: delayValEl,
  format: v => Math.round(v),
  onCommit: v => postConfig({ delayMs: Math.round(v) })
});

UI.initTabs?.(document);

postConfig(config);

window.addEventListener('unhandledrejection', event => {
  const reason = event?.reason;
  const message = typeof reason === 'string' ? reason : reason?.message;
  if (message) {
    showFatalError(`Fatal runtime error: ${message}`, reason);
  } else {
    showFatalError('An unexpected runtime error occurred.', reason);
  }
  event?.preventDefault?.();
});
