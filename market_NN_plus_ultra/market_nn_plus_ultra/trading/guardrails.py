from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, Mapping

import pandas as pd

from ..evaluation import guardrail_metrics


@dataclass(slots=True)
class GuardrailConfig:
    """Configuration describing guardrail enforcement thresholds."""

    enabled: bool = False
    capital_base: float = 1.0
    tail_percentile: float = 5.0
    max_gross_exposure: float | None = None
    max_net_exposure: float | None = None
    max_turnover: float | None = None
    min_tail_return: float | None = None
    max_tail_frequency: float | None = None
    max_symbol_exposure: float | None = None
    sector_caps: Mapping[str, float] = field(default_factory=dict)
    sector_column: str = "sector"
    factor_caps: Mapping[str, float] = field(default_factory=dict)
    factor_column: str = "factor"
    timestamp_col: str = "timestamp"
    symbol_col: str = "symbol"
    notional_col: str = "notional"
    position_col: str = "position"
    price_col: str = "price"
    return_col: str = "pnl"
    enforcement: str = "clip"

    def serialisable(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation of the configuration."""

        payload = asdict(self)
        payload["sector_caps"] = {key: float(value) for key, value in self.sector_caps.items()}
        payload["factor_caps"] = {key: float(value) for key, value in self.factor_caps.items()}
        return payload


@dataclass(slots=True)
class GuardrailViolation:
    """Record describing a guardrail breach after enforcement."""

    name: str
    value: float
    threshold: float | None
    message: str


@dataclass(slots=True)
class GuardrailResult:
    """Outcome of evaluating a trade log against the guardrails."""

    trades: pd.DataFrame
    metrics: Dict[str, float]
    violations: list[GuardrailViolation]
    scaled: bool = False
    exposures: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def as_payload(self) -> Dict[str, Any]:
        """Serialise the result for JSON responses."""

        exposures_payload = {
            group: {name: float(value) for name, value in values.items()}
            for group, values in self.exposures.items()
        }
        return {
            "metrics": dict(self.metrics),
            "violations": [asdict(v) for v in self.violations],
            "scaled": self.scaled,
            "trades": self.trades.to_dict(orient="records"),
            "exposures": exposures_payload,
        }


class GuardrailPolicy:
    """Apply exposure, turnover, and tail-risk guardrails to trade logs."""

    def __init__(self, config: GuardrailConfig) -> None:
        self.config = config
        self._clip_enabled = config.enforcement.lower() == "clip"

    def enforce(self, trades: pd.DataFrame) -> GuardrailResult:
        """Apply guardrails to ``trades`` and return the resulting metrics."""

        if not isinstance(trades, pd.DataFrame):
            raise TypeError("trades must be a pandas DataFrame")

        df = trades.copy(deep=True)
        cfg = self.config

        if cfg.timestamp_col not in df.columns:
            raise ValueError(
                f"Column '{cfg.timestamp_col}' is required for guardrail evaluation"
            )

        if cfg.notional_col not in df.columns:
            if cfg.position_col not in df.columns or cfg.price_col not in df.columns:
                raise ValueError(
                    "Guardrail evaluation requires either a notional column or both "
                    f"position ('{cfg.position_col}') and price ('{cfg.price_col}') columns"
                )
            df[cfg.notional_col] = df[cfg.position_col] * df[cfg.price_col]

        df = df.sort_values(cfg.timestamp_col).reset_index(drop=True)

        metrics = guardrail_metrics(
            df,
            timestamp_col=cfg.timestamp_col,
            symbol_col=cfg.symbol_col,
            notional_col=cfg.notional_col,
            position_col=cfg.position_col,
            price_col=cfg.price_col,
            return_col=cfg.return_col,
            capital_base=cfg.capital_base,
            tail_percentile=cfg.tail_percentile,
        )

        def _capital() -> float:
            return cfg.capital_base if cfg.capital_base not in (None, 0) else 1.0

        def _compute_exposures() -> Dict[str, Dict[str, float]]:
            capital = _capital()
            exposures: Dict[str, Dict[str, float]] = {}

            def _summarise(label: str, column: str | None) -> None:
                if not column or column not in df.columns:
                    return
                series = (
                    df.groupby(column)[cfg.notional_col]
                    .apply(lambda values: values.abs().max())
                    .dropna()
                )
                if series.empty:
                    exposures[label] = {}
                    return
                normalised = (series / capital).sort_values(ascending=False)
                exposures[label] = {
                    str(key): float(value)
                    for key, value in normalised.items()
                }

            _summarise("symbol", cfg.symbol_col)
            _summarise("sector", cfg.sector_column)
            _summarise("factor", cfg.factor_column)
            return {label: values for label, values in exposures.items() if values or label == "symbol"}

        if not cfg.enabled:
            exposures = _compute_exposures()
            return GuardrailResult(
                trades=df,
                metrics=metrics,
                violations=[],
                scaled=False,
                exposures=exposures,
            )

        scaled = False

        def _scale_all(scale: float) -> None:
            nonlocal df, metrics, scaled
            if not self._clip_enabled or scale >= 1.0:
                return
            scale = float(max(scale, 0.0))
            if scale == 1.0:
                return
            df[cfg.notional_col] = df[cfg.notional_col] * scale
            if cfg.position_col in df.columns:
                df[cfg.position_col] = df[cfg.position_col] * scale
            if cfg.return_col in df.columns:
                df[cfg.return_col] = df[cfg.return_col] * scale
            scaled = True
            metrics = guardrail_metrics(
                df,
                timestamp_col=cfg.timestamp_col,
                symbol_col=cfg.symbol_col,
                notional_col=cfg.notional_col,
                position_col=cfg.position_col,
                price_col=cfg.price_col,
                return_col=cfg.return_col,
                capital_base=cfg.capital_base,
                tail_percentile=cfg.tail_percentile,
            )

        def _scale_group(column: str, selections: Iterable[Any], limits: Mapping[str, float]) -> None:
            nonlocal metrics, scaled
            if column not in df.columns:
                raise ValueError(
                    f"Guardrail configuration references column '{column}' which is missing from trades"
                )
            capital = _capital()
            notional = df[cfg.notional_col].abs()
            grouped = notional.groupby(df[column]).max() / capital
            for key in selections:
                if key not in grouped:
                    continue
                exposure = float(grouped[key])
                limit = float(limits[key])
                if exposure <= limit or limit <= 0:
                    continue
                if not self._clip_enabled:
                    continue
                ratio = limit / exposure
                if ratio >= 1.0:
                    continue
                mask = df[column] == key
                df.loc[mask, cfg.notional_col] = df.loc[mask, cfg.notional_col] * ratio
                if cfg.position_col in df.columns:
                    df.loc[mask, cfg.position_col] = df.loc[mask, cfg.position_col] * ratio
                if cfg.return_col in df.columns:
                    df.loc[mask, cfg.return_col] = df.loc[mask, cfg.return_col] * ratio
                scaled = True
            if scaled:
                metrics = guardrail_metrics(
                    df,
                    timestamp_col=cfg.timestamp_col,
                    symbol_col=cfg.symbol_col,
                    notional_col=cfg.notional_col,
                    position_col=cfg.position_col,
                    price_col=cfg.price_col,
                    return_col=cfg.return_col,
                    capital_base=cfg.capital_base,
                    tail_percentile=cfg.tail_percentile,
                )

        # Enforce per-symbol exposure caps before global adjustments.
        symbol_limit = cfg.max_symbol_exposure
        if symbol_limit is not None and symbol_limit > 0:
            if cfg.symbol_col not in df.columns:
                raise ValueError(
                    f"Symbol column '{cfg.symbol_col}' is required for symbol exposure guardrails"
                )
            capital = _capital()
            notional = df[cfg.notional_col].abs()
            by_symbol = notional.groupby(df[cfg.symbol_col]).max()
            if not by_symbol.empty:
                exposures = by_symbol / capital
                if self._clip_enabled:
                    for symbol, exposure in exposures.items():
                        if exposure <= symbol_limit:
                            continue
                        ratio = symbol_limit / float(exposure) if exposure > 0 else 0.0
                        if ratio < 1.0:
                            mask = df[cfg.symbol_col] == symbol
                            df.loc[mask, cfg.notional_col] = df.loc[mask, cfg.notional_col] * ratio
                            if cfg.position_col in df.columns:
                                df.loc[mask, cfg.position_col] = df.loc[mask, cfg.position_col] * ratio
                            if cfg.return_col in df.columns:
                                df.loc[mask, cfg.return_col] = df.loc[mask, cfg.return_col] * ratio
                            scaled = True
                    if scaled:
                        metrics = guardrail_metrics(
                            df,
                            timestamp_col=cfg.timestamp_col,
                            symbol_col=cfg.symbol_col,
                            notional_col=cfg.notional_col,
                            position_col=cfg.position_col,
                            price_col=cfg.price_col,
                            return_col=cfg.return_col,
                            capital_base=cfg.capital_base,
                            tail_percentile=cfg.tail_percentile,
                        )

        # Sector and factor caps
        if cfg.sector_caps:
            _scale_group(cfg.sector_column, cfg.sector_caps.keys(), cfg.sector_caps)
        if cfg.factor_caps:
            _scale_group(cfg.factor_column, cfg.factor_caps.keys(), cfg.factor_caps)

        gross_limit = cfg.max_gross_exposure
        if gross_limit is not None and gross_limit > 0:
            gross_value = float(metrics.get("gross_exposure_peak", 0.0))
            if gross_value > gross_limit and self._clip_enabled:
                ratio = gross_limit / gross_value if gross_value > 0 else 0.0
                _scale_all(ratio)

        net_limit = cfg.max_net_exposure
        if net_limit is not None and net_limit > 0:
            net_value = float(metrics.get("net_exposure_peak", 0.0))
            if net_value > net_limit and self._clip_enabled:
                ratio = net_limit / net_value if net_value > 0 else 0.0
                _scale_all(ratio)

        turnover_limit = cfg.max_turnover
        if turnover_limit is not None and turnover_limit > 0:
            turnover_value = float(metrics.get("turnover_rate", 0.0))
            if turnover_value > turnover_limit and self._clip_enabled:
                ratio = turnover_limit / turnover_value if turnover_value > 0 else 0.0
                _scale_all(ratio)

        tail_limit = cfg.min_tail_return
        if tail_limit is not None:
            tail_value = float(metrics.get("tail_return_quantile", 0.0))
            if tail_value < tail_limit and self._clip_enabled and tail_value != 0:
                ratio = tail_limit / tail_value
                if ratio > 0 and ratio < 1:
                    _scale_all(ratio)

        # Final metrics after enforcement
        metrics = guardrail_metrics(
            df,
            timestamp_col=cfg.timestamp_col,
            symbol_col=cfg.symbol_col,
            notional_col=cfg.notional_col,
            position_col=cfg.position_col,
            price_col=cfg.price_col,
            return_col=cfg.return_col,
            capital_base=cfg.capital_base,
            tail_percentile=cfg.tail_percentile,
        )
        exposures = _compute_exposures()

        violations: list[GuardrailViolation] = []

        def _record_violation(name: str, value: float, threshold: float | None, message: str) -> None:
            violations.append(
                GuardrailViolation(
                    name=name,
                    value=float(value),
                    threshold=float(threshold) if threshold is not None else None,
                    message=message,
                )
            )

        gross_value = float(metrics.get("gross_exposure_peak", 0.0))
        if gross_limit is not None and gross_value > gross_limit:
            _record_violation(
                "gross_exposure_peak",
                gross_value,
                gross_limit,
                f"Gross exposure peak {gross_value:.3f} exceeded limit {gross_limit:.3f}",
            )

        net_value = float(metrics.get("net_exposure_peak", 0.0))
        if net_limit is not None and net_value > net_limit:
            _record_violation(
                "net_exposure_peak",
                net_value,
                net_limit,
                f"Net exposure peak {net_value:.3f} exceeded limit {net_limit:.3f}",
            )

        turnover_value = float(metrics.get("turnover_rate", 0.0))
        if turnover_limit is not None and turnover_value > turnover_limit:
            _record_violation(
                "turnover_rate",
                turnover_value,
                turnover_limit,
                f"Turnover rate {turnover_value:.3f} exceeded limit {turnover_limit:.3f}",
            )

        tail_value = float(metrics.get("tail_return_quantile", 0.0))
        if tail_limit is not None and tail_value < tail_limit:
            _record_violation(
                "tail_return_quantile",
                tail_value,
                tail_limit,
                f"Tail return quantile {tail_value:.3f} breached floor {tail_limit:.3f}",
            )

        tail_freq_limit = cfg.max_tail_frequency
        tail_freq_value = float(metrics.get("tail_event_frequency", 0.0))
        if tail_freq_limit is not None and tail_freq_value > tail_freq_limit:
            _record_violation(
                "tail_event_frequency",
                tail_freq_value,
                tail_freq_limit,
                f"Tail event frequency {tail_freq_value:.3f} exceeded limit {tail_freq_limit:.3f}",
            )

        if cfg.max_symbol_exposure is not None and cfg.max_symbol_exposure > 0:
            symbol_exposures = exposures.get("symbol", {})
            for symbol, exposure in symbol_exposures.items():
                if exposure > cfg.max_symbol_exposure:
                    _record_violation(
                        f"symbol_exposure:{symbol}",
                        float(exposure),
                        cfg.max_symbol_exposure,
                        f"Symbol {symbol} exposure {float(exposure):.3f} exceeded limit {cfg.max_symbol_exposure:.3f}",
                    )

        if cfg.sector_caps:
            if cfg.sector_column not in df.columns:
                _record_violation(
                    "sector_caps",
                    0.0,
                    None,
                    f"Sector column '{cfg.sector_column}' missing for configured caps",
                )
            else:
                sector_exposure = exposures.get("sector", {})
                for sector, limit in cfg.sector_caps.items():
                    exposure = float(sector_exposure.get(sector, 0.0))
                    if exposure > limit:
                        _record_violation(
                            f"sector_exposure:{sector}",
                            exposure,
                            limit,
                            f"Sector {sector} exposure {exposure:.3f} exceeded limit {float(limit):.3f}",
                        )

        if cfg.factor_caps:
            if cfg.factor_column not in df.columns:
                _record_violation(
                    "factor_caps",
                    0.0,
                    None,
                    f"Factor column '{cfg.factor_column}' missing for configured caps",
                )
            else:
                factor_exposure = exposures.get("factor", {})
                for factor, limit in cfg.factor_caps.items():
                    exposure = float(factor_exposure.get(factor, 0.0))
                    if exposure > limit:
                        _record_violation(
                            f"factor_exposure:{factor}",
                            exposure,
                            limit,
                            f"Factor {factor} exposure {exposure:.3f} exceeded limit {float(limit):.3f}",
                        )

        return GuardrailResult(
            trades=df,
            metrics=metrics,
            violations=violations,
            scaled=scaled,
            exposures=exposures,
        )


__all__ = [
    "GuardrailConfig",
    "GuardrailPolicy",
    "GuardrailResult",
    "GuardrailViolation",
]
