(() => {
  const UI = window.NeuralUI;

  const priceCanvas = document.getElementById('priceChart');
  const priceCtx = priceCanvas.getContext('2d');
  const errorCanvas = document.getElementById('errorChart');
  const errorCtx = errorCanvas.getContext('2d');
  const weightsCanvas = document.getElementById('weightsCanvas');
  const weightsCtx = weightsCanvas.getContext('2d');
  const outputCanvas = document.getElementById('outputCanvas');
  const outputCtx = outputCanvas.getContext('2d');

  const latestActualEl = document.getElementById('latestActual');
  const latestPredEl = document.getElementById('latestPred');
  const latestErrorEl = document.getElementById('latestError');
  const datasetProgressEl = document.getElementById('datasetProgress');
  const windowSizeLabelEl = document.getElementById('windowSizeLabel');
  const hiddenUnitsLabelEl = document.getElementById('hiddenUnitsLabel');
  const stepCountEl = document.getElementById('stepCount');
  const maeEl = document.getElementById('mae');
  const rmseEl = document.getElementById('rmse');
  const bestMaeEl = document.getElementById('bestMae');
  const lrLabelEl = document.getElementById('lrLabel');
  const noiseLabelEl = document.getElementById('noiseLabel');
  const pointsStreamedEl = document.getElementById('pointsStreamed');
  const windowCoverageEl = document.getElementById('windowCoverage');
  const loopCountEl = document.getElementById('loopCount');
  const bestMaeSecondaryEl = document.getElementById('bestMaeSecondary');
  const lastResetEl = document.getElementById('lastReset');
  const instrumentLabelEl = document.getElementById('instrumentLabel');
  const datasetDateRangeEl = document.getElementById('datasetDateRange');
  const datasetPlaylistEl = document.getElementById('datasetPlaylist');
  const datasetUniverseEl = document.getElementById('datasetUniverse');
  const recentListEl = document.getElementById('recentList');
  const weightsRawEl = document.getElementById('weightsRaw');
  const portfolioEquityEl = document.getElementById('portfolioEquity');
  const portfolioCashEl = document.getElementById('portfolioCash');
  const portfolioPositionEl = document.getElementById('portfolioPosition');
  const portfolioAvgCostEl = document.getElementById('portfolioAvgCost');
  const portfolioUnrealizedEl = document.getElementById('portfolioUnrealized');
  const portfolioReturnEl = document.getElementById('portfolioReturn');
  const portfolioRealizedEl = document.getElementById('portfolioRealized');
  const portfolioDrawdownEl = document.getElementById('portfolioDrawdown');
  const portfolioSharpeEl = document.getElementById('portfolioSharpe');
  const portfolioSortinoEl = document.getElementById('portfolioSortino');
  const portfolioCalmarEl = document.getElementById('portfolioCalmar');
  const portfolioCagrEl = document.getElementById('portfolioCagr');
  const portfolioDownsideEl = document.getElementById('portfolioDownside');
  const portfolioTurnoverEl = document.getElementById('portfolioTurnover');
  const portfolioAvgWinLossEl = document.getElementById('portfolioAvgWinLoss');
  const portfolioAvgHoldEl = document.getElementById('portfolioAvgHold');
  const portfolioCostDragEl = document.getElementById('portfolioCostDrag');
  const portfolioCapacityEl = document.getElementById('portfolioCapacity');
  const portfolioTradeCountEl = document.getElementById('portfolioTradeCount');
  const portfolioWinRateEl = document.getElementById('portfolioWinRate');
  const portfolioCostsPaidEl = document.getElementById('portfolioCostsPaid');
  const tradeLogEl = document.getElementById('tradeLog');
  const tradingLastActionEl = document.getElementById('tradingLastAction');
  const tradingConfidenceEl = document.getElementById('tradingConfidence');
  const tradingEdgeEl = document.getElementById('tradingEdge');
  const tradingLastRewardEl = document.getElementById('tradingLastReward');
  const tradingAvgRewardEl = document.getElementById('tradingAvgReward');
  const tradingExplorationEl = document.getElementById('tradingExploration');
  const tradingCycleCountEl = document.getElementById('tradingCycleCount');
  const tradingLastCycleEl = document.getElementById('tradingLastCycle');
  const tradingBestCycleEl = document.getElementById('tradingBestCycle');
  const tradingLifetimeRewardEl = document.getElementById('tradingLifetimeReward');
  const tradingCumulativeReturnEl = document.getElementById('tradingCumulativeReturn');
  const tradingTradeCountEl = document.getElementById('tradingTradeCount');
  const tradingWinRateEl = document.getElementById('tradingWinRate');
  const tradingRegimeEl = document.getElementById('tradingRegime');
  const tradingVolatilityEl = document.getElementById('tradingVolatility');
  const tradingRiskEl = document.getElementById('tradingRisk');
  const tradingThresholdEl = document.getElementById('tradingThreshold');

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

  function showFatalError(message, error) {
    if (error) {
      console.error(message, error);
    } else {
      console.error(message);
    }
    fatalErrorEl.textContent = message;
    fatalErrorEl.style.display = 'block';
    fatalError = true;
    startBtn.disabled = true;
    pauseBtn.disabled = true;
    resetBtn.disabled = true;
  }

  const config = {
    learningRate: parseFloat(learningRateEl.value),
    hiddenUnits: parseInt(hiddenUnitsEl.value, 10),
    windowSize: parseInt(windowSizeEl.value, 10),
    noise: parseFloat(noiseEl.value),
    delayMs: parseInt(delayEl.value, 10)
  };

  let running = false;
  let worker = null;
  let fatalError = false;

  function formatNumber(value, digits = 2) {
    if (value == null || Number.isNaN(value)) return '—';
    return Number(value).toFixed(digits);
  }

  function formatCurrency(value) {
    if (!Number.isFinite(value)) return '—';
    return `$${Number(value).toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    })}`;
  }

  function formatPercent(value, digits = 2) {
    if (!Number.isFinite(value)) return '—';
    return `${(value * 100).toFixed(digits)}%`;
  }

  function formatShares(value) {
    if (!Number.isFinite(value)) return '—';
    return `${Number(value).toLocaleString(undefined, {
      maximumFractionDigits: 0
    })} sh`;
  }

  function formatInteger(value) {
    if (!Number.isFinite(value)) return '—';
    return Number(value).toLocaleString();
  }

  function formatSigned(value, digits = 3) {
    if (value == null || Number.isNaN(value)) return '—';
    const abs = Math.abs(value).toFixed(digits);
    const sign = value > 0 ? '+' : value < 0 ? '−' : '';
    return `${sign}${abs}`;
  }

  function prettifyReason(reason) {
    if (!reason) return null;
    const normalized = String(reason).replace(/_/g, ' ').trim();
    if (!normalized) return null;
    return normalized.replace(/\b\w/g, char => char.toUpperCase());
  }

  function drawPriceChart(actual, predicted) {
    const w = priceCanvas.width;
    const h = priceCanvas.height;
    priceCtx.clearRect(0, 0, w, h);
    if (!actual?.length) return;
    const total = actual.length;
    const margin = 16;
    const minVal = Math.min(...actual, ...(predicted || actual));
    const maxVal = Math.max(...actual, ...(predicted || actual));
    const span = Math.max(1e-6, maxVal - minVal);

    priceCtx.fillStyle = '#fbfbfb';
    priceCtx.fillRect(0, 0, w, h);
    priceCtx.strokeStyle = '#e0e0e0';
    priceCtx.lineWidth = 1;
    const gridLines = 4;
    for (let i = 0; i <= gridLines; i++) {
      const y = margin + ((h - margin * 2) * i) / gridLines;
      priceCtx.beginPath();
      priceCtx.moveTo(margin, y);
      priceCtx.lineTo(w - margin, y);
      priceCtx.stroke();
    }

    const drawSeries = (series, color) => {
      if (!series?.length) return;
      priceCtx.strokeStyle = color;
      priceCtx.lineWidth = 2;
      priceCtx.beginPath();
      series.forEach((value, idx) => {
        const x = margin + ((w - margin * 2) * idx) / Math.max(1, total - 1);
        const y = h - margin - ((value - minVal) / span) * (h - margin * 2);
        if (idx === 0) priceCtx.moveTo(x, y);
        else priceCtx.lineTo(x, y);
      });
      priceCtx.stroke();
    };

    drawSeries(actual, '#ff922b');
    drawSeries(predicted, '#4dabf7');
  }

  function drawErrorChart(errors) {
    const w = errorCanvas.width;
    const h = errorCanvas.height;
    errorCtx.clearRect(0, 0, w, h);
    if (!errors?.length) return;
    const margin = 12;
    const total = errors.length;
    const maxVal = Math.max(...errors, 1e-6);
    errorCtx.fillStyle = '#fbfbfb';
    errorCtx.fillRect(0, 0, w, h);
    errorCtx.strokeStyle = '#e5e5e5';
    errorCtx.lineWidth = 1;
    const gridLines = 4;
    for (let i = 0; i <= gridLines; i++) {
      const y = margin + ((h - margin * 2) * i) / gridLines;
      errorCtx.beginPath();
      errorCtx.moveTo(margin, y);
      errorCtx.lineTo(w - margin, y);
      errorCtx.stroke();
    }

    errorCtx.strokeStyle = '#748ffc';
    errorCtx.lineWidth = 2;
    errorCtx.beginPath();
    errors.forEach((value, idx) => {
      const x = margin + ((w - margin * 2) * idx) / Math.max(1, total - 1);
      const y = h - margin - (value / maxVal) * (h - margin * 2);
      if (idx === 0) errorCtx.moveTo(x, y);
      else errorCtx.lineTo(x, y);
    });
    errorCtx.stroke();
  }

  function drawWeightsHeatmap(weights, hiddenUnits, inputSize) {
    const w = weightsCanvas.width;
    const h = weightsCanvas.height;
    weightsCtx.clearRect(0, 0, w, h);
    if (!weights || !hiddenUnits || !inputSize) return;
    const cellW = w / inputSize;
    const cellH = h / hiddenUnits;
    const maxAbs = Math.max(...weights.map(v => Math.abs(v)), 1e-6);
    for (let row = 0; row < hiddenUnits; row++) {
      for (let col = 0; col < inputSize; col++) {
        const value = weights[row * inputSize + col];
        const norm = Math.abs(value) / maxAbs;
        const alpha = 0.15 + 0.85 * norm;
        weightsCtx.fillStyle = value >= 0 ? `rgba(255, 146, 43, ${alpha.toFixed(3)})` : `rgba(21, 170, 191, ${alpha.toFixed(3)})`;
        weightsCtx.fillRect(col * cellW, row * cellH, cellW + 1, cellH + 1);
      }
    }
    weightsCtx.strokeStyle = 'rgba(0,0,0,0.15)';
    weightsCtx.lineWidth = 1;
    for (let row = 0; row <= hiddenUnits; row++) {
      const y = row * cellH;
      weightsCtx.beginPath();
      weightsCtx.moveTo(0, y);
      weightsCtx.lineTo(w, y);
      weightsCtx.stroke();
    }
    for (let col = 0; col <= inputSize; col++) {
      const x = col * cellW;
      weightsCtx.beginPath();
      weightsCtx.moveTo(x, 0);
      weightsCtx.lineTo(x, h);
      weightsCtx.stroke();
    }
  }

  function drawOutputWeights(weights) {
    const w = outputCanvas.width;
    const h = outputCanvas.height;
    outputCtx.clearRect(0, 0, w, h);
    if (!weights?.length) return;
    const maxAbs = Math.max(...weights.map(v => Math.abs(v)), 1e-6);
    const midY = h / 2;
    outputCtx.fillStyle = '#fbfbfb';
    outputCtx.fillRect(0, 0, w, h);
    outputCtx.strokeStyle = '#bbbbbb';
    outputCtx.beginPath();
    outputCtx.moveTo(0, midY);
    outputCtx.lineTo(w, midY);
    outputCtx.stroke();
    const barHeight = h / weights.length;
    weights.forEach((value, idx) => {
      const norm = value / maxAbs;
      const barWidth = (w / 2) * Math.abs(norm);
      const y = idx * barHeight;
      outputCtx.fillStyle = value >= 0 ? 'rgba(255, 146, 43, 0.85)' : 'rgba(21, 170, 191, 0.85)';
      if (value >= 0) {
        outputCtx.fillRect(w / 2, y + 2, barWidth, barHeight - 4);
      } else {
        outputCtx.fillRect(w / 2 - barWidth, y + 2, barWidth, barHeight - 4);
      }
    });
  }

  function updateRecentList(entries) {
    recentListEl.innerHTML = '';
    if (!entries?.length) {
      const placeholder = document.createElement('li');
      placeholder.textContent = 'Waiting for predictions…';
      recentListEl.appendChild(placeholder);
      return;
    }
    entries.forEach(item => {
      const li = document.createElement('li');
      const label = document.createElement('span');
      label.textContent = item.label || `t-${item.offset ?? 0}`;
      const actual = document.createElement('span');
      actual.textContent = `${formatNumber(item.actual, 2)}`;
      const pred = document.createElement('span');
      pred.textContent = `${formatNumber(item.predicted, 2)}`;
      pred.classList.add('prediction');
      const error = document.createElement('span');
      const diff = item.error;
      error.textContent = `Δ ${formatSigned(diff, 3)}`;
      error.classList.add('error');
      if (diff > 0) error.classList.add('over');
      else if (diff < 0) error.classList.add('under');
      li.appendChild(label);
      li.appendChild(actual);
      li.appendChild(pred);
      li.appendChild(error);
      recentListEl.appendChild(li);
    });
  }

  function updateWeightsRaw(snapshot) {
    if (!snapshot?.weights) {
      weightsRawEl.textContent = '—';
      return;
    }
    const { inputWeights, outputWeights } = snapshot.weights;
    const parts = [];
    if (inputWeights) {
      parts.push(`Input → hidden (${inputWeights.length} values)\n${JSON.stringify(inputWeights.slice(0, 120))}${inputWeights.length > 120 ? '…' : ''}`);
    }
    if (outputWeights) {
      parts.push(`Hidden → output (${outputWeights.length} values)\n${JSON.stringify(outputWeights)}`);
    }
    weightsRawEl.textContent = parts.join('\n\n');
  }

  function updateTradeLog(trades) {
    tradeLogEl.innerHTML = '';
    if (!trades?.length) {
      const placeholder = document.createElement('li');
      placeholder.textContent = 'No trades placed yet — the policy network is still exploring the tape.';
      tradeLogEl.appendChild(placeholder);
      return;
    }

    trades.forEach(trade => {
      const li = document.createElement('li');

      const dateSpan = document.createElement('span');
      dateSpan.textContent = trade.date || '—';

      const sideSpan = document.createElement('span');
      sideSpan.textContent = trade.side;
      sideSpan.classList.add('side', trade.side?.toLowerCase?.() || '');

      const sizeSpan = document.createElement('span');
      const shares = Number(trade.shares ?? 0);
      sizeSpan.textContent = `${shares.toLocaleString(undefined, { maximumFractionDigits: 0 })} @ ${formatNumber(trade.price, 2)}`;

      const metaSpan = document.createElement('span');
      const edgeText = Number.isFinite(trade.edgePct)
        ? `edge ${formatSigned(trade.edgePct, 2)}%`
        : null;
      const pnlText = Number.isFinite(trade.pnl)
        ? `P/L ${formatCurrency(trade.pnl)}`
        : null;
      const costText = Number.isFinite(trade.costs) && trade.costs !== 0
        ? `cost ${formatCurrency(trade.costs)}`
        : null;
      const holdText = Number.isFinite(trade.holdingDays) && trade.holdingDays > 0
        ? `${trade.holdingDays.toFixed(1)}d hold`
        : null;
      const metaParts = [];
      if (edgeText) metaParts.push(edgeText);
      if (pnlText) metaParts.push(pnlText);
      if (costText) metaParts.push(costText);
      if (holdText) metaParts.push(holdText);
      metaSpan.textContent = metaParts.join(' · ');
      if (Number.isFinite(trade.pnl)) {
        metaSpan.classList.add('pnl');
        if (trade.pnl > 0) metaSpan.classList.add('gain');
        else if (trade.pnl < 0) metaSpan.classList.add('loss');
      }

      li.appendChild(dateSpan);
      li.appendChild(sideSpan);
      li.appendChild(sizeSpan);
      li.appendChild(metaSpan);
      tradeLogEl.appendChild(li);
    });
  }

  function updatePortfolioCard(portfolio) {
    if (!portfolio) {
      portfolioEquityEl.textContent = '—';
      portfolioCashEl.textContent = '—';
      portfolioPositionEl.textContent = '—';
    portfolioAvgCostEl.textContent = '—';
    portfolioUnrealizedEl.textContent = '—';
    portfolioReturnEl.textContent = '—';
    portfolioRealizedEl.textContent = '—';
    portfolioDrawdownEl.textContent = '—';
    portfolioSharpeEl.textContent = '—';
    if (portfolioSortinoEl) portfolioSortinoEl.textContent = '—';
    if (portfolioCalmarEl) portfolioCalmarEl.textContent = '—';
    if (portfolioCagrEl) portfolioCagrEl.textContent = '—';
    if (portfolioDownsideEl) portfolioDownsideEl.textContent = '—';
    if (portfolioTurnoverEl) portfolioTurnoverEl.textContent = '—';
    if (portfolioAvgWinLossEl) portfolioAvgWinLossEl.textContent = '—';
    if (portfolioAvgHoldEl) portfolioAvgHoldEl.textContent = '—';
    if (portfolioCostDragEl) portfolioCostDragEl.textContent = '—';
    if (portfolioCapacityEl) portfolioCapacityEl.textContent = '—';
    portfolioTradeCountEl.textContent = '—';
    portfolioWinRateEl.textContent = '—';
    if (portfolioCostsPaidEl) portfolioCostsPaidEl.textContent = '—';
    updateTradeLog([]);
    updateTradingSummary(null);
    return;
  }

    portfolioEquityEl.textContent = formatCurrency(portfolio.equity);
    portfolioCashEl.textContent = formatCurrency(portfolio.cash);
    portfolioPositionEl.textContent = formatShares(portfolio.position);
    portfolioAvgCostEl.textContent = formatCurrency(portfolio.avgCost);
    portfolioUnrealizedEl.textContent = formatCurrency(portfolio.unrealizedPnl);
    portfolioReturnEl.textContent = formatPercent(portfolio.totalReturn);
    portfolioRealizedEl.textContent = formatCurrency(portfolio.realizedPnl);
    portfolioDrawdownEl.textContent = Number.isFinite(portfolio.maxDrawdown)
      ? formatPercent(portfolio.maxDrawdown, 2)
      : '—';
    portfolioSharpeEl.textContent = Number.isFinite(portfolio.sharpe)
      ? formatNumber(portfolio.sharpe, 2)
      : '—';
    const closedTrades = portfolio.closedTradeCount ?? 0;
    if (portfolioSortinoEl) {
      portfolioSortinoEl.textContent = Number.isFinite(portfolio.sortino)
        ? formatNumber(portfolio.sortino, 2)
        : '—';
    }
    if (portfolioCalmarEl) {
      portfolioCalmarEl.textContent = Number.isFinite(portfolio.calmar)
        ? formatNumber(portfolio.calmar, 2)
        : '—';
    }
    if (portfolioCagrEl) {
      portfolioCagrEl.textContent = Number.isFinite(portfolio.cagr)
        ? formatPercent(portfolio.cagr, 2)
        : '—';
    }
    if (portfolioDownsideEl) {
      portfolioDownsideEl.textContent = Number.isFinite(portfolio.downsideDeviation)
        ? formatPercent(portfolio.downsideDeviation, 2)
        : '—';
    }
    if (portfolioTurnoverEl) {
      portfolioTurnoverEl.textContent = Number.isFinite(portfolio.turnover)
        ? formatPercent(portfolio.turnover, 1)
        : '—';
    }
    if (portfolioAvgWinLossEl) {
      const avgWin = portfolio.avgWin;
      const avgLoss = portfolio.avgLoss;
      const hasWin = Number.isFinite(avgWin);
      const hasLoss = Number.isFinite(avgLoss);
      if (hasWin || hasLoss) {
        const winText = hasWin ? formatCurrency(avgWin) : '—';
        const lossText = hasLoss ? formatCurrency(avgLoss) : '—';
        let text = `${winText} / ${lossText}`;
        if (Number.isFinite(portfolio.winLossRatio)) {
          text += ` (R=${formatNumber(portfolio.winLossRatio, 2)})`;
        }
        portfolioAvgWinLossEl.textContent = text;
      } else {
        portfolioAvgWinLossEl.textContent = '—';
      }
    }
    if (portfolioAvgHoldEl) {
      portfolioAvgHoldEl.textContent = closedTrades > 0 && Number.isFinite(portfolio.avgHoldDays)
        ? `${portfolio.avgHoldDays.toFixed(1)} d`
        : '—';
    }
    if (portfolioCostDragEl) {
      portfolioCostDragEl.textContent = Number.isFinite(portfolio.costDrag)
        ? formatPercent(portfolio.costDrag, 2)
        : '—';
    }
    if (portfolioCapacityEl) {
      portfolioCapacityEl.textContent = Number.isFinite(portfolio.capacityPerShare)
        ? formatCurrency(portfolio.capacityPerShare)
        : '—';
    }
    portfolioTradeCountEl.textContent = formatInteger(portfolio.tradeCount ?? 0);
    portfolioWinRateEl.textContent = closedTrades > 0
      ? formatPercent(portfolio.winRate, 2)
      : '—';
    if (portfolioCostsPaidEl) {
      portfolioCostsPaidEl.textContent = formatCurrency(portfolio.costsPaid ?? 0);
    }
    updateTradeLog(portfolio.trades);
  }

  function applyPolicyClass(el, value) {
    if (!el) return;
    el.classList.remove('gain', 'loss');
    if (!Number.isFinite(value)) return;
    if (value > 0) el.classList.add('gain');
    else if (value < 0) el.classList.add('loss');
  }

  function updateTradingSummary(trading) {
    if (!trading) {
      tradingLastActionEl.textContent = '—';
      tradingConfidenceEl.textContent = '—';
      tradingEdgeEl.textContent = '—';
      tradingLastRewardEl.textContent = '—';
      tradingAvgRewardEl.textContent = '—';
      tradingExplorationEl.textContent = '—';
      tradingCycleCountEl.textContent = '—';
      tradingLastCycleEl.textContent = '—';
      tradingBestCycleEl.textContent = '—';
      tradingLifetimeRewardEl.textContent = '—';
      tradingCumulativeReturnEl.textContent = '—';
      tradingTradeCountEl.textContent = '—';
      tradingWinRateEl.textContent = '—';
      if (tradingRegimeEl) tradingRegimeEl.textContent = '—';
      if (tradingVolatilityEl) tradingVolatilityEl.textContent = '—';
      if (tradingRiskEl) tradingRiskEl.textContent = '—';
      if (tradingThresholdEl) tradingThresholdEl.textContent = '—';
      applyPolicyClass(tradingLastRewardEl, null);
      applyPolicyClass(tradingAvgRewardEl, null);
      applyPolicyClass(tradingEdgeEl, null);
      applyPolicyClass(tradingLastCycleEl, null);
      applyPolicyClass(tradingBestCycleEl, null);
      applyPolicyClass(tradingRegimeEl, null);
      applyPolicyClass(tradingVolatilityEl, null);
      applyPolicyClass(tradingRiskEl, null);
      return;
    }

    const haveSamples = (trading.steps ?? 0) > 0;
    tradingLastActionEl.textContent = haveSamples ? (trading.lastAction ?? '—') : '—';
    tradingConfidenceEl.textContent = haveSamples && Number.isFinite(trading.lastConfidence)
      ? formatPercent(trading.lastConfidence, 1)
      : '—';
    tradingEdgeEl.textContent = haveSamples && Number.isFinite(trading.lastEdge)
      ? formatPercent(trading.lastEdge, 2)
      : '—';
    tradingLastRewardEl.textContent = haveSamples && Number.isFinite(trading.lastReward)
      ? formatSigned(trading.lastReward, 4)
      : '—';
    tradingAvgRewardEl.textContent = haveSamples && Number.isFinite(trading.avgReward)
      ? formatSigned(trading.avgReward, 4)
      : '—';
    tradingExplorationEl.textContent = Number.isFinite(trading.exploration)
      ? formatPercent(trading.exploration, 1)
      : '—';
    tradingCycleCountEl.textContent = formatInteger(trading.cycleCount ?? 0);
    tradingLastCycleEl.textContent = Number.isFinite(trading.lastCycleReturn)
      ? formatPercent(trading.lastCycleReturn, 2)
      : '—';
    tradingBestCycleEl.textContent = Number.isFinite(trading.bestCycleReturn)
      ? formatPercent(trading.bestCycleReturn, 2)
      : '—';
    tradingLifetimeRewardEl.textContent = formatSigned(trading.lifetimeReward, 4);
    tradingCumulativeReturnEl.textContent = Number.isFinite(trading.cumulativeReturn)
      ? formatPercent(trading.cumulativeReturn, 2)
      : '—';
    tradingTradeCountEl.textContent = formatInteger(trading.trades ?? 0);
    tradingWinRateEl.textContent = haveSamples && Number.isFinite(trading.winRate)
      ? formatPercent(trading.winRate, 2)
      : '—';
    if (tradingRegimeEl) {
      if (haveSamples && Number.isFinite(trading.regime)) {
        const trend = trading.regime;
        const label = trend > 0.1 ? 'Bull' : trend < -0.1 ? 'Bear' : 'Flat';
        tradingRegimeEl.textContent = `${label} (${formatSigned(trend, 2)})`;
        applyPolicyClass(tradingRegimeEl, trend);
      } else {
        tradingRegimeEl.textContent = '—';
        applyPolicyClass(tradingRegimeEl, null);
      }
    }
    if (tradingVolatilityEl) {
      if (haveSamples && (Number.isFinite(trading.realizedVol) || Number.isFinite(trading.volZ))) {
        const volParts = [];
        if (Number.isFinite(trading.realizedVol)) {
          volParts.push(`σ=${formatPercent(trading.realizedVol, 2)}`);
        }
        if (Number.isFinite(trading.volZ)) {
          volParts.push(`z=${formatSigned(trading.volZ, 2)}`);
        }
        const bucketLabel = trading.volBucket > 0 ? 'High' : trading.volBucket < 0 ? 'Low' : 'Normal';
        volParts.push(bucketLabel);
        tradingVolatilityEl.textContent = volParts.join(' · ');
        applyPolicyClass(tradingVolatilityEl, -trading.volBucket);
      } else {
        tradingVolatilityEl.textContent = '—';
        applyPolicyClass(tradingVolatilityEl, null);
      }
    }
    if (tradingRiskEl) {
      if (haveSamples) {
        const parts = [];
        const cooldown = Math.max(0, Math.round(trading.cooldown ?? 0));
        if (cooldown > 0) {
          const reasonText = prettifyReason(trading.cooldownReason);
          const reason = reasonText ? ` (${reasonText})` : '';
          parts.push(`Cooldown ${cooldown} bars${reason}`);
        } else {
          parts.push('Active');
        }
        if (cooldown <= 0 && trading.gateReason) {
          const gateText = prettifyReason(trading.gateReason);
          if (gateText) {
            parts.push(`Gate: ${gateText}`);
          }
        }
        if (Number.isFinite(trading.maxExposure)) {
          parts.push(`Max exp ${formatPercent(trading.maxExposure, 1)}`);
        }
        if (Number.isFinite(trading.targetVol)) {
          parts.push(`Target σ ${formatPercent(trading.targetVol, 1)}`);
        }
        tradingRiskEl.textContent = parts.join(' · ');
        applyPolicyClass(tradingRiskEl, cooldown > 0 ? -1 : 1);
      } else {
        tradingRiskEl.textContent = '—';
        applyPolicyClass(tradingRiskEl, null);
      }
    }
    if (tradingThresholdEl) {
      if (haveSamples) {
        const thresholds = [];
        if (Number.isFinite(trading.minEdge)) {
          thresholds.push(`edge ≥ ${formatPercent(trading.minEdge, 2)}`);
        }
        if (Number.isFinite(trading.trendThreshold)) {
          thresholds.push(`trend ≥ ${formatSigned(trading.trendThreshold, 2)}`);
        }
        if (Number.isFinite(trading.volatilityCap)) {
          thresholds.push(`σ ≤ ${formatPercent(trading.volatilityCap, 1)}`);
        }
        if (Number.isFinite(trading.volZCap)) {
          thresholds.push(`|z| ≤ ${formatNumber(trading.volZCap, 2)}`);
        }
        tradingThresholdEl.textContent = thresholds.join(' · ');
      } else {
        tradingThresholdEl.textContent = '—';
      }
    }
    applyPolicyClass(tradingLastRewardEl, haveSamples ? trading.lastReward : null);
    applyPolicyClass(tradingAvgRewardEl, haveSamples ? trading.avgReward : null);
    applyPolicyClass(tradingEdgeEl, Number.isFinite(trading.lastEdge) ? trading.lastEdge : null);
    applyPolicyClass(tradingLastCycleEl, Number.isFinite(trading.lastCycleReturn) ? trading.lastCycleReturn : null);
    applyPolicyClass(tradingBestCycleEl, Number.isFinite(trading.bestCycleReturn) ? trading.bestCycleReturn : null);
  }

  function applySnapshot(snapshot) {
    if (!snapshot) return;
    const { history, stats, weights } = snapshot;
    drawPriceChart(history?.actual, history?.predicted);
    drawErrorChart(history?.errors);
    drawWeightsHeatmap(weights?.inputWeights, weights?.hiddenUnits, weights?.inputSize);
    drawOutputWeights(weights?.outputWeights);
    updateWeightsRaw(snapshot);
    updateRecentList(snapshot.recentPredictions);

    latestActualEl.textContent = formatNumber(stats?.lastActual, 2);
    latestPredEl.textContent = formatNumber(stats?.lastPredicted, 2);
    latestErrorEl.textContent = formatNumber(stats?.lastAbsError, 3);
    datasetProgressEl.textContent = stats?.progressPct != null ? `${formatNumber(stats.progressPct, 1)}%` : '—';
    windowSizeLabelEl.textContent = stats?.windowSize ?? '—';
    hiddenUnitsLabelEl.textContent = stats?.hiddenUnits ?? '—';
    stepCountEl.textContent = formatInteger(stats?.steps);
    maeEl.textContent = formatNumber(stats?.mae, 4);
    rmseEl.textContent = formatNumber(stats?.rmse, 4);
    bestMaeEl.textContent = formatNumber(stats?.bestMae, 4);
    lrLabelEl.textContent = formatNumber(stats?.learningRate, 3);
    noiseLabelEl.textContent = formatNumber(stats?.noise, 3);
    pointsStreamedEl.textContent = formatInteger(stats?.pointsSeen);
    windowCoverageEl.textContent = stats?.windowCoverage ?? '—';
    loopCountEl.textContent = formatInteger(stats?.loops);
    bestMaeSecondaryEl.textContent = formatNumber(stats?.bestMae, 4);
    lastResetEl.textContent = stats?.lastReset ?? '—';
    if (instrumentLabelEl) {
      const ticker = stats?.ticker;
      const instrumentName = stats?.instrumentName;
      if (ticker && instrumentName) {
        instrumentLabelEl.textContent = `${ticker} — ${instrumentName}`;
      } else if (instrumentName) {
        instrumentLabelEl.textContent = instrumentName;
      } else if (ticker) {
        instrumentLabelEl.textContent = ticker;
      } else {
        instrumentLabelEl.textContent = '—';
      }
    }
    if (datasetDateRangeEl) datasetDateRangeEl.textContent = stats?.dateRange ?? '—';
    if (datasetPlaylistEl) {
      const position = stats?.playlistPosition ?? 0;
      const size = stats?.playlistSize ?? 0;
      datasetPlaylistEl.textContent = size > 0 ? `${position}/${size}` : '—';
    }
    if (datasetUniverseEl) datasetUniverseEl.textContent = formatInteger(stats?.availableTickers);

    updatePortfolioCard(snapshot.portfolio);
    updateTradingSummary(snapshot.trading);
  }

  function setRunningState(value) {
    running = value;
    if (fatalError) {
      startBtn.disabled = true;
      pauseBtn.disabled = true;
      return;
    }
    startBtn.disabled = running;
    pauseBtn.disabled = !running;
  }

  function resolveWorkerUrl() {
    const candidates = [];
    const scriptEl = document.currentScript;
    if (scriptEl?.src) candidates.push(scriptEl.src);
    // Fallback to any script tag that looks like our own bundle.
    const fallbackScript = document.querySelector('script[src$="main.js"]');
    if (fallbackScript?.src) candidates.push(fallbackScript.src);
    if (document.baseURI) candidates.push(document.baseURI);
    if (window.location?.href) candidates.push(window.location.href);

    for (const base of candidates) {
      try {
        if (!base) continue;
        const resolved = new URL('worker.js', base);
        if (resolved?.href) {
          return resolved.href;
        }
      } catch (error) {
        console.warn('Unable to resolve worker URL from base', base, error);
      }
    }

    try {
      const origin = window.location?.origin;
      if (origin) {
        const fallback = new URL('worker.js', `${origin}${window.location?.pathname || '/'}`);
        if (fallback?.href) {
          return fallback.href;
        }
      }
    } catch (error) {
      console.warn('Unable to resolve worker URL from window location', error);
    }

    console.warn('Falling back to relative worker script path');
    return './worker.js';
  }

  function initWorker() {
    if (typeof Worker !== 'function') {
      showFatalError('This browser does not support Web Workers, which are required for this experience.');
      return null;
    }
    try {
      const workerUrl = resolveWorkerUrl();
      const instance = new Worker(workerUrl, { type: 'classic' });
      instance.onmessage = ev => {
        const msg = ev.data;
        if (!msg) return;
        if (msg.type === 'status') {
          setRunningState(!!msg.running);
        } else if (msg.type === 'snapshot') {
          applySnapshot(msg.snapshot);
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
      showFatalError('Unable to start the background worker. This experience cannot run.', error);
      return null;
    }
  }

  worker = initWorker();

  function postConfig(update) {
    Object.assign(config, update);
    if (worker) {
      worker.postMessage({ type: 'config', config: { ...config } });
    }
  }

  startBtn.addEventListener('click', () => {
    if (worker) {
      worker.postMessage({ type: 'start' });
    }
  });

  pauseBtn.addEventListener('click', () => {
    if (worker) {
      worker.postMessage({ type: 'pause' });
    }
  });

  resetBtn.addEventListener('click', () => {
    if (worker) {
      worker.postMessage({ type: 'reset' });
    }
  });

  UI.bindRangeControl(learningRateEl, {
    valueEl: learningRateValEl,
    format: v => v.toFixed(3),
    onCommit: v => postConfig({ learningRate: v })
  });

  UI.bindRangeControl(hiddenUnitsEl, {
    valueEl: hiddenUnitsValEl,
    format: v => Math.round(v),
    onCommit: v => postConfig({ hiddenUnits: Math.round(v) })
  });

  UI.bindRangeControl(windowSizeEl, {
    valueEl: windowSizeValEl,
    format: v => Math.round(v),
    onCommit: v => postConfig({ windowSize: Math.round(v) })
  });

  UI.bindRangeControl(noiseEl, {
    valueEl: noiseValEl,
    format: v => v.toFixed(3),
    onCommit: v => postConfig({ noise: v })
  });

  UI.bindRangeControl(delayEl, {
    valueEl: delayValEl,
    format: v => Math.round(v),
    onCommit: v => postConfig({ delayMs: Math.round(v) })
  });

  UI.initTabs(document);

  // Prime the worker with the initial configuration.
  postConfig({});
})();
