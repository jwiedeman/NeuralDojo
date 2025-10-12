import {
  formatCurrency,
  formatInteger,
  formatNumber,
  formatPercent,
  formatShares,
  formatSigned,
  prettifyReason
} from './formatters.js';

function safeContext(canvas, type = '2d') {
  if (!canvas) return null;
  try {
    return canvas.getContext(type);
  } catch (error) {
    console.warn('Failed to acquire canvas context', { canvas, type }, error);
    return null;
  }
}

export class DashboardView {
  constructor(elements) {
    this.el = elements;
    this.priceCanvas = elements.priceCanvas;
    this.errorCanvas = elements.errorCanvas;
    this.weightsCanvas = elements.weightsCanvas;
    this.outputCanvas = elements.outputCanvas;

    this.priceCtx = safeContext(this.priceCanvas);
    this.errorCtx = safeContext(this.errorCanvas);
    this.weightsCtx = safeContext(this.weightsCanvas);
    this.outputCtx = safeContext(this.outputCanvas);
  }

  drawPriceChart(actual, predicted) {
    if (!this.priceCanvas || !this.priceCtx) return;
    const ctx = this.priceCtx;
    const w = this.priceCanvas.width;
    const h = this.priceCanvas.height;
    ctx.clearRect(0, 0, w, h);
    if (!actual?.length) return;
    const total = actual.length;
    const margin = 16;
    const reference = predicted?.length ? predicted : actual;
    const minVal = Math.min(...actual, ...reference);
    const maxVal = Math.max(...actual, ...reference);
    const span = Math.max(1e-6, maxVal - minVal);

    ctx.fillStyle = '#fbfbfb';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    const gridLines = 4;
    for (let i = 0; i <= gridLines; i++) {
      const y = margin + ((h - margin * 2) * i) / gridLines;
      ctx.beginPath();
      ctx.moveTo(margin, y);
      ctx.lineTo(w - margin, y);
      ctx.stroke();
    }

    const drawSeries = (series, color) => {
      if (!series?.length) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      series.forEach((value, idx) => {
        const x = margin + ((w - margin * 2) * idx) / Math.max(1, total - 1);
        const y = h - margin - ((value - minVal) / span) * (h - margin * 2);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };

    drawSeries(actual, '#ff922b');
    drawSeries(predicted, '#4dabf7');
  }

  drawErrorChart(errors) {
    if (!this.errorCanvas || !this.errorCtx) return;
    const ctx = this.errorCtx;
    const w = this.errorCanvas.width;
    const h = this.errorCanvas.height;
    ctx.clearRect(0, 0, w, h);
    if (!errors?.length) return;
    const margin = 12;
    const total = errors.length;
    const maxVal = Math.max(...errors, 1e-6);

    ctx.fillStyle = '#fbfbfb';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = '#e5e5e5';
    ctx.lineWidth = 1;
    const gridLines = 4;
    for (let i = 0; i <= gridLines; i++) {
      const y = margin + ((h - margin * 2) * i) / gridLines;
      ctx.beginPath();
      ctx.moveTo(margin, y);
      ctx.lineTo(w - margin, y);
      ctx.stroke();
    }

    ctx.strokeStyle = '#748ffc';
    ctx.lineWidth = 2;
    ctx.beginPath();
    errors.forEach((value, idx) => {
      const x = margin + ((w - margin * 2) * idx) / Math.max(1, total - 1);
      const y = h - margin - (value / maxVal) * (h - margin * 2);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  drawWeightsHeatmap(weights, hiddenUnits, inputSize) {
    if (!this.weightsCanvas || !this.weightsCtx) return;
    const ctx = this.weightsCtx;
    const w = this.weightsCanvas.width;
    const h = this.weightsCanvas.height;
    ctx.clearRect(0, 0, w, h);
    if (!weights || !hiddenUnits || !inputSize) return;
    const cellW = w / inputSize;
    const cellH = h / hiddenUnits;
    const maxAbs = Math.max(...weights.map(v => Math.abs(v)), 1e-6);

    for (let row = 0; row < hiddenUnits; row++) {
      for (let col = 0; col < inputSize; col++) {
        const value = weights[row * inputSize + col];
        const norm = Math.abs(value) / maxAbs;
        const alpha = 0.15 + 0.85 * norm;
        ctx.fillStyle = value >= 0
          ? `rgba(255, 146, 43, ${alpha.toFixed(3)})`
          : `rgba(21, 170, 191, ${alpha.toFixed(3)})`;
        ctx.fillRect(col * cellW, row * cellH, cellW + 1, cellH + 1);
      }
    }

    ctx.strokeStyle = 'rgba(0,0,0,0.15)';
    ctx.lineWidth = 1;
    for (let row = 0; row <= hiddenUnits; row++) {
      const y = row * cellH;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
    for (let col = 0; col <= inputSize; col++) {
      const x = col * cellW;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
  }

  drawOutputWeights(weights) {
    if (!this.outputCanvas || !this.outputCtx) return;
    const ctx = this.outputCtx;
    const w = this.outputCanvas.width;
    const h = this.outputCanvas.height;
    ctx.clearRect(0, 0, w, h);
    if (!weights?.length) return;
    const maxAbs = Math.max(...weights.map(v => Math.abs(v)), 1e-6);
    const midY = h / 2;

    ctx.fillStyle = '#fbfbfb';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = '#bbbbbb';
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(w, midY);
    ctx.stroke();

    const barHeight = h / weights.length;
    weights.forEach((value, idx) => {
      const norm = value / maxAbs;
      const barWidth = (w / 2) * Math.abs(norm);
      const y = idx * barHeight;
      ctx.fillStyle = value >= 0 ? 'rgba(255, 146, 43, 0.85)' : 'rgba(21, 170, 191, 0.85)';
      if (value >= 0) {
        ctx.fillRect(w / 2, y + 2, barWidth, barHeight - 4);
      } else {
        ctx.fillRect(w / 2 - barWidth, y + 2, barWidth, barHeight - 4);
      }
    });
  }

  updateRecentList(entries) {
    const listEl = this.el.recentListEl;
    if (!listEl) return;
    listEl.innerHTML = '';
    if (!entries?.length) {
      const placeholder = document.createElement('li');
      placeholder.textContent = 'Waiting for predictions…';
      listEl.appendChild(placeholder);
      return;
    }

    entries.forEach(item => {
      const li = document.createElement('li');
      const label = document.createElement('span');
      label.className = 'label';
      const details = document.createElement('span');
      details.className = 'details';
      const errorSpan = document.createElement('span');
      errorSpan.className = 'error';

      const ticker = item?.ticker || item?.symbol || '';
      const when = item?.date ? ` · ${item.date}` : '';
      label.textContent = ticker ? `${ticker}${when}` : item?.date || 'Unknown';

      const actual = formatNumber(item?.actual, 2);
      const predicted = formatNumber(item?.predicted, 2);
      details.textContent = `actual ${actual} → forecast ${predicted}`;

      const absError = Number.isFinite(item?.absError) ? Math.abs(item.absError) : null;
      if (Number.isFinite(absError)) {
        errorSpan.textContent = `err ${formatNumber(absError, 4)}`;
        if (item?.direction === 'over') errorSpan.classList.add('over');
        else if (item?.direction === 'under') errorSpan.classList.add('under');
        else errorSpan.classList.add('even');
        if (absError < 0.1) errorSpan.classList.add('low');
        else if (absError > 0.5) errorSpan.classList.add('bad');
      } else {
        errorSpan.textContent = 'err —';
        errorSpan.classList.add('even');
      }

      li.appendChild(label);
      li.appendChild(details);
      li.appendChild(errorSpan);
      listEl.appendChild(li);
    });
  }

  updateTradeLog(trades) {
    const logEl = this.el.tradeLogEl;
    if (!logEl) return;
    logEl.innerHTML = '';
    if (!trades?.length) {
      const placeholder = document.createElement('li');
      placeholder.textContent = 'No trades executed yet.';
      logEl.appendChild(placeholder);
      return;
    }

    trades.slice(-25).reverse().forEach(trade => {
      const li = document.createElement('li');
      const dateSpan = document.createElement('span');
      dateSpan.className = 'trade-date';
      dateSpan.textContent = trade?.date || '—';

      const sideSpan = document.createElement('span');
      sideSpan.className = `trade-side ${trade?.side || 'neutral'}`;
      sideSpan.textContent = trade?.side ? trade.side.toUpperCase() : '—';

      const sizeSpan = document.createElement('span');
      sizeSpan.className = 'trade-size';
      sizeSpan.textContent = Number.isFinite(trade?.size)
        ? formatShares(trade.size)
        : '—';

      const metaSpan = document.createElement('span');
      metaSpan.className = 'trade-meta';
      const priceText = Number.isFinite(trade?.price)
        ? formatCurrency(trade.price)
        : '—';
      const pnlText = Number.isFinite(trade?.pnl)
        ? formatCurrency(trade.pnl)
        : '—';
      const retText = Number.isFinite(trade?.return)
        ? formatPercent(trade.return, 2)
        : '—';
      metaSpan.textContent = `${priceText} · ${pnlText} · ${retText}`;
      if (Number.isFinite(trade?.pnl)) {
        metaSpan.classList.add('pnl');
        if (trade.pnl > 0) metaSpan.classList.add('gain');
        else if (trade.pnl < 0) metaSpan.classList.add('loss');
      }

      li.appendChild(dateSpan);
      li.appendChild(sideSpan);
      li.appendChild(sizeSpan);
      li.appendChild(metaSpan);
      logEl.appendChild(li);
    });
  }

  applyPolicyClass(el, value) {
    if (!el) return;
    el.classList.remove('positive', 'negative', 'neutral');
    if (!Number.isFinite(value)) {
      el.classList.add('neutral');
      return;
    }
    if (value > 0.0001) {
      el.classList.add('positive');
    } else if (value < -0.0001) {
      el.classList.add('negative');
    } else {
      el.classList.add('neutral');
    }
  }

  updateTradingSummary(trading) {
    const {
      tradingLastActionEl,
      tradingConfidenceEl,
      tradingEdgeEl,
      tradingLastRewardEl,
      tradingAvgRewardEl,
      tradingExplorationEl,
      tradingCycleCountEl,
      tradingLastCycleEl,
      tradingBestCycleEl,
      tradingLifetimeRewardEl,
      tradingCumulativeReturnEl,
      tradingTradeCountEl,
      tradingWinRateEl,
      tradingRegimeEl,
      tradingVolatilityEl,
      tradingRiskEl,
      tradingThresholdEl
    } = this.el;

    if (!trading) {
      if (tradingLastActionEl) tradingLastActionEl.textContent = '—';
      if (tradingConfidenceEl) tradingConfidenceEl.textContent = '—';
      if (tradingEdgeEl) tradingEdgeEl.textContent = '—';
      if (tradingLastRewardEl) tradingLastRewardEl.textContent = '—';
      if (tradingAvgRewardEl) tradingAvgRewardEl.textContent = '—';
      if (tradingExplorationEl) tradingExplorationEl.textContent = '—';
      if (tradingCycleCountEl) tradingCycleCountEl.textContent = '0';
      if (tradingLastCycleEl) tradingLastCycleEl.textContent = '—';
      if (tradingBestCycleEl) tradingBestCycleEl.textContent = '—';
      if (tradingLifetimeRewardEl) tradingLifetimeRewardEl.textContent = '—';
      if (tradingCumulativeReturnEl) tradingCumulativeReturnEl.textContent = '—';
      if (tradingTradeCountEl) tradingTradeCountEl.textContent = '0';
      if (tradingWinRateEl) tradingWinRateEl.textContent = '—';
      if (tradingRegimeEl) tradingRegimeEl.textContent = '—';
      if (tradingVolatilityEl) tradingVolatilityEl.textContent = '—';
      if (tradingRiskEl) tradingRiskEl.textContent = '—';
      if (tradingThresholdEl) tradingThresholdEl.textContent = '—';
      return;
    }

    const haveSamples = (trading?.samples ?? 0) > 0;
    if (tradingLastActionEl) tradingLastActionEl.textContent = trading.lastAction ?? '—';
    if (tradingConfidenceEl) {
      tradingConfidenceEl.textContent = Number.isFinite(trading.confidence)
        ? `${(trading.confidence * 100).toFixed(1)}%`
        : '—';
      this.applyPolicyClass(tradingConfidenceEl, trading.confidence);
    }
    if (tradingEdgeEl) {
      tradingEdgeEl.textContent = Number.isFinite(trading.lastEdge)
        ? formatPercent(trading.lastEdge, 2)
        : '—';
      this.applyPolicyClass(tradingEdgeEl, trading.lastEdge);
    }
    if (tradingLastRewardEl) {
      tradingLastRewardEl.textContent = Number.isFinite(trading.lastReward)
        ? formatSigned(trading.lastReward, 3)
        : '—';
      this.applyPolicyClass(tradingLastRewardEl, trading.lastReward);
    }
    if (tradingAvgRewardEl) {
      tradingAvgRewardEl.textContent = Number.isFinite(trading.avgReward)
        ? formatSigned(trading.avgReward, 3)
        : '—';
      this.applyPolicyClass(tradingAvgRewardEl, trading.avgReward);
    }
    if (tradingExplorationEl) {
      tradingExplorationEl.textContent = Number.isFinite(trading.exploration)
        ? `${(trading.exploration * 100).toFixed(1)}%`
        : '—';
    }
    if (tradingCycleCountEl) {
      tradingCycleCountEl.textContent = formatInteger(trading.cycleCount ?? 0);
    }
    if (tradingLastCycleEl) {
      tradingLastCycleEl.textContent = Number.isFinite(trading.lastCycleReturn)
        ? formatPercent(trading.lastCycleReturn, 2)
        : '—';
      this.applyPolicyClass(tradingLastCycleEl, trading.lastCycleReturn);
    }
    if (tradingBestCycleEl) {
      tradingBestCycleEl.textContent = Number.isFinite(trading.bestCycleReturn)
        ? formatPercent(trading.bestCycleReturn, 2)
        : '—';
      this.applyPolicyClass(tradingBestCycleEl, trading.bestCycleReturn);
    }
    if (tradingLifetimeRewardEl) {
      tradingLifetimeRewardEl.textContent = Number.isFinite(trading.lifetimeReward)
        ? formatSigned(trading.lifetimeReward, 3)
        : '—';
    }
    if (tradingCumulativeReturnEl) {
      tradingCumulativeReturnEl.textContent = Number.isFinite(trading.cumulativeReturn)
        ? formatPercent(trading.cumulativeReturn, 2)
        : '—';
    }
    if (tradingTradeCountEl) {
      tradingTradeCountEl.textContent = formatInteger(trading.tradeCount ?? 0);
    }
    if (tradingWinRateEl) {
      tradingWinRateEl.textContent = haveSamples && Number.isFinite(trading.winRate)
        ? formatPercent(trading.winRate, 2)
        : '—';
    }
    if (tradingRegimeEl) {
      if (haveSamples && Number.isFinite(trading.trend)) {
        const trend = trading.trend;
        const direction = trend > 0 ? 'Uptrend' : trend < 0 ? 'Downtrend' : 'Sideways';
        tradingRegimeEl.textContent = `${direction} (${formatSigned(trend, 2)})`;
        this.applyPolicyClass(tradingRegimeEl, trend);
      } else {
        tradingRegimeEl.textContent = '—';
        this.applyPolicyClass(tradingRegimeEl, null);
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
        this.applyPolicyClass(tradingVolatilityEl, -trading.volBucket);
      } else {
        tradingVolatilityEl.textContent = '—';
        this.applyPolicyClass(tradingVolatilityEl, null);
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
        this.applyPolicyClass(tradingRiskEl, cooldown > 0 ? -1 : 1);
      } else {
        tradingRiskEl.textContent = '—';
        this.applyPolicyClass(tradingRiskEl, null);
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

    this.applyPolicyClass(tradingLastRewardEl, haveSamples ? trading.lastReward : null);
    this.applyPolicyClass(tradingAvgRewardEl, haveSamples ? trading.avgReward : null);
    this.applyPolicyClass(tradingEdgeEl, Number.isFinite(trading.lastEdge) ? trading.lastEdge : null);
    this.applyPolicyClass(tradingLastCycleEl, Number.isFinite(trading.lastCycleReturn) ? trading.lastCycleReturn : null);
    this.applyPolicyClass(tradingBestCycleEl, Number.isFinite(trading.bestCycleReturn) ? trading.bestCycleReturn : null);
  }

  updatePortfolioCard(portfolio) {
    const {
      portfolioEquityEl,
      portfolioCashEl,
      portfolioPositionEl,
      portfolioAvgCostEl,
      portfolioUnrealizedEl,
      portfolioReturnEl,
      portfolioRealizedEl,
      portfolioDrawdownEl,
      portfolioSharpeEl,
      portfolioSortinoEl,
      portfolioCalmarEl,
      portfolioCagrEl,
      portfolioDownsideEl,
      portfolioTurnoverEl,
      portfolioAvgWinLossEl,
      portfolioAvgHoldEl,
      portfolioCostDragEl,
      portfolioCapacityEl,
      portfolioTradeCountEl,
      portfolioWinRateEl,
      portfolioCostsPaidEl
    } = this.el;

    if (!portfolio) {
      if (portfolioEquityEl) portfolioEquityEl.textContent = '—';
      if (portfolioCashEl) portfolioCashEl.textContent = '—';
      if (portfolioPositionEl) portfolioPositionEl.textContent = '—';
      if (portfolioAvgCostEl) portfolioAvgCostEl.textContent = '—';
      if (portfolioUnrealizedEl) portfolioUnrealizedEl.textContent = '—';
      if (portfolioReturnEl) portfolioReturnEl.textContent = '—';
      if (portfolioRealizedEl) portfolioRealizedEl.textContent = '—';
      if (portfolioDrawdownEl) portfolioDrawdownEl.textContent = '—';
      if (portfolioSharpeEl) portfolioSharpeEl.textContent = '—';
      if (portfolioSortinoEl) portfolioSortinoEl.textContent = '—';
      if (portfolioCalmarEl) portfolioCalmarEl.textContent = '—';
      if (portfolioCagrEl) portfolioCagrEl.textContent = '—';
      if (portfolioDownsideEl) portfolioDownsideEl.textContent = '—';
      if (portfolioTurnoverEl) portfolioTurnoverEl.textContent = '—';
      if (portfolioAvgWinLossEl) portfolioAvgWinLossEl.textContent = '—';
      if (portfolioAvgHoldEl) portfolioAvgHoldEl.textContent = '—';
      if (portfolioCostDragEl) portfolioCostDragEl.textContent = '—';
      if (portfolioCapacityEl) portfolioCapacityEl.textContent = '—';
      if (portfolioTradeCountEl) portfolioTradeCountEl.textContent = '—';
      if (portfolioWinRateEl) portfolioWinRateEl.textContent = '—';
      if (portfolioCostsPaidEl) portfolioCostsPaidEl.textContent = '—';
      this.updateTradeLog([]);
      this.updateTradingSummary(null);
      return;
    }

    if (portfolioEquityEl) portfolioEquityEl.textContent = formatCurrency(portfolio.equity);
    if (portfolioCashEl) portfolioCashEl.textContent = formatCurrency(portfolio.cash);
    if (portfolioPositionEl) portfolioPositionEl.textContent = formatShares(portfolio.position);
    if (portfolioAvgCostEl) portfolioAvgCostEl.textContent = formatCurrency(portfolio.avgCost);
    if (portfolioUnrealizedEl) portfolioUnrealizedEl.textContent = formatCurrency(portfolio.unrealizedPnl);
    if (portfolioReturnEl) portfolioReturnEl.textContent = formatPercent(portfolio.totalReturn);
    if (portfolioRealizedEl) portfolioRealizedEl.textContent = formatCurrency(portfolio.realizedPnl);
    if (portfolioDrawdownEl) {
      portfolioDrawdownEl.textContent = Number.isFinite(portfolio.maxDrawdown)
        ? formatPercent(portfolio.maxDrawdown, 2)
        : '—';
    }
    if (portfolioSharpeEl) {
      portfolioSharpeEl.textContent = Number.isFinite(portfolio.sharpe)
        ? formatNumber(portfolio.sharpe, 2)
        : '—';
    }
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
    if (portfolioTradeCountEl) portfolioTradeCountEl.textContent = formatInteger(portfolio.tradeCount ?? 0);
    if (portfolioWinRateEl) {
      portfolioWinRateEl.textContent = closedTrades > 0
        ? formatPercent(portfolio.winRate, 2)
        : '—';
    }
    if (portfolioCostsPaidEl) {
      portfolioCostsPaidEl.textContent = formatCurrency(portfolio.costsPaid ?? 0);
    }
    this.updateTradeLog(portfolio.trades);
  }

  updateWeightsRaw(snapshot) {
    const el = this.el.weightsRawEl;
    if (!el) return;
    if (!snapshot?.weights) {
      el.textContent = '—';
      return;
    }
    const { weights } = snapshot;
    const stringify = obj => JSON.stringify(obj, null, 2);
    const sections = [];
    if (weights.inputWeights) {
      sections.push(`Input → Hidden\n${stringify(weights.inputWeights)}`);
    }
    if (weights.outputWeights) {
      sections.push(`Hidden → Output\n${stringify(weights.outputWeights)}`);
    }
    if (weights.biases) {
      sections.push(`Biases\n${stringify(weights.biases)}`);
    }
    el.textContent = sections.join('\n\n');
  }

  updateForwardProjection(forecast) {
    const body = this.el.forwardTableBodyEl;
    const emptyEl = this.el.forwardEmptyEl;
    if (!body) return;
    body.innerHTML = '';
    const hasForecast = forecast && Array.isArray(forecast.predicted) && forecast.predicted.length;
    if (!hasForecast) {
      if (emptyEl) emptyEl.style.display = 'block';
      return;
    }
    if (emptyEl) emptyEl.style.display = 'none';
    const predictedList = Array.isArray(forecast.predicted) ? forecast.predicted : [];
    const actualList = Array.isArray(forecast.actual) ? forecast.actual : [];
    const dateList = Array.isArray(forecast.dates) ? forecast.dates : [];
    for (let i = 0; i < predictedList.length; i++) {
      const row = document.createElement('tr');

      const stepCell = document.createElement('td');
      stepCell.className = 'step';
      stepCell.textContent = `${i + 1}`;

      const dateCell = document.createElement('td');
      dateCell.className = 'date';
      dateCell.textContent = dateList[i] ?? '—';

      const forecastCell = document.createElement('td');
      forecastCell.className = 'forecast';
      const predictedValue = predictedList[i];
      forecastCell.textContent = Number.isFinite(predictedValue)
        ? formatNumber(predictedValue, 2)
        : '—';

      const actualCell = document.createElement('td');
      actualCell.className = 'actual';
      const actualValue = actualList[i];
      actualCell.textContent = Number.isFinite(actualValue)
        ? formatNumber(actualValue, 2)
        : '—';

      const deltaCell = document.createElement('td');
      deltaCell.className = 'delta';
      const diff = Number.isFinite(actualValue) && Number.isFinite(predictedValue)
        ? actualValue - predictedValue
        : null;
      if (Number.isFinite(diff)) {
        deltaCell.textContent = formatSigned(diff, 2);
        if (diff > 0.0001) deltaCell.classList.add('positive');
        else if (diff < -0.0001) deltaCell.classList.add('negative');
        else deltaCell.classList.add('neutral');
      } else {
        deltaCell.textContent = '—';
        deltaCell.classList.add('neutral');
      }

      const pctCell = document.createElement('td');
      pctCell.className = 'delta-pct';
      const pct = Number.isFinite(diff) && Number.isFinite(actualValue) && Math.abs(actualValue) > 1e-6
        ? diff / actualValue
        : null;
      if (Number.isFinite(pct)) {
        pctCell.textContent = formatPercent(pct, 2);
        if (pct > 0.0001) pctCell.classList.add('positive');
        else if (pct < -0.0001) pctCell.classList.add('negative');
        else pctCell.classList.add('neutral');
      } else {
        pctCell.textContent = '—';
        pctCell.classList.add('neutral');
      }

      row.appendChild(stepCell);
      row.appendChild(dateCell);
      row.appendChild(forecastCell);
      row.appendChild(actualCell);
      row.appendChild(deltaCell);
      row.appendChild(pctCell);
      body.appendChild(row);
    }
  }

  applySnapshot(snapshot) {
    if (!snapshot) return;
    const { history, stats, weights } = snapshot;
    this.drawPriceChart(history?.actual, history?.predicted);
    this.drawErrorChart(history?.errors);
    this.drawWeightsHeatmap(weights?.inputWeights, weights?.hiddenUnits, weights?.inputSize);
    this.drawOutputWeights(weights?.outputWeights);
    this.updateWeightsRaw(snapshot);
    this.updateRecentList(snapshot.recentPredictions);
    this.updateForwardProjection(snapshot.forecast);

    const {
      latestActualEl,
      latestPredEl,
      latestErrorEl,
      datasetProgressEl,
      windowSizeLabelEl,
      hiddenUnitsLabelEl,
      stepCountEl,
      maeEl,
      rmseEl,
      bestMaeEl,
      lrLabelEl,
      noiseLabelEl,
      pointsStreamedEl,
      windowCoverageEl,
      loopCountEl,
      bestMaeSecondaryEl,
      lastResetEl,
      instrumentLabelEl,
      datasetDateRangeEl,
      datasetPlaylistEl,
      datasetUniverseEl,
      forecastHorizonLabelEl
    } = this.el;

    if (latestActualEl) latestActualEl.textContent = formatNumber(stats?.lastActual, 2);
    if (latestPredEl) latestPredEl.textContent = formatNumber(stats?.lastPredicted, 2);
    if (latestErrorEl) latestErrorEl.textContent = formatNumber(stats?.lastAbsError, 3);
    if (datasetProgressEl) {
      datasetProgressEl.textContent = stats?.progressPct != null
        ? `${formatNumber(stats.progressPct, 1)}%`
        : '—';
    }
    if (windowSizeLabelEl) windowSizeLabelEl.textContent = stats?.windowSize ?? '—';
    if (hiddenUnitsLabelEl) hiddenUnitsLabelEl.textContent = stats?.hiddenUnits ?? '—';
    if (stepCountEl) stepCountEl.textContent = formatInteger(stats?.steps);
    if (maeEl) maeEl.textContent = formatNumber(stats?.mae, 4);
    if (rmseEl) rmseEl.textContent = formatNumber(stats?.rmse, 4);
    if (bestMaeEl) bestMaeEl.textContent = formatNumber(stats?.bestMae, 4);
    if (lrLabelEl) lrLabelEl.textContent = formatNumber(stats?.learningRate, 3);
    if (noiseLabelEl) noiseLabelEl.textContent = formatNumber(stats?.noise, 3);
    if (pointsStreamedEl) pointsStreamedEl.textContent = formatInteger(stats?.pointsSeen);
    if (windowCoverageEl) windowCoverageEl.textContent = stats?.windowCoverage ?? '—';
    if (loopCountEl) loopCountEl.textContent = formatInteger(stats?.loops);
    if (bestMaeSecondaryEl) bestMaeSecondaryEl.textContent = formatNumber(stats?.bestMae, 4);
    if (lastResetEl) lastResetEl.textContent = stats?.lastReset ?? '—';

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
    if (forecastHorizonLabelEl) {
      const horizon = stats?.forecastHorizon;
      forecastHorizonLabelEl.textContent = Number.isFinite(horizon)
        ? `${horizon} sessions`
        : '—';
    }

    this.updatePortfolioCard(snapshot.portfolio);
    this.updateTradingSummary(snapshot.trading);
  }
}
