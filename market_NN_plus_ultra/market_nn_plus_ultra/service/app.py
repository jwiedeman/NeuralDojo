"""FastAPI application exposing Market NN Plus Ultra inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import math
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from ..trading import AgentRunResult, MarketNNPlusUltraAgent
from ..training import ExperimentConfig, load_experiment_from_file, summarise_curriculum_profile


@dataclass(slots=True)
class ServiceSettings:
    """Static configuration used to boot the inference service."""

    config_path: Path
    checkpoint_path: Optional[Path] = None
    device: str = "cpu"
    evaluate_by_default: bool = True
    return_column: str = "realised_return"
    max_prediction_rows: Optional[int] = 1024

    def __post_init__(self) -> None:
        self.config_path = Path(self.config_path)
        if self.checkpoint_path is not None:
            self.checkpoint_path = Path(self.checkpoint_path)


class PredictionRequest(BaseModel):
    """Request payload accepted by the prediction endpoint."""

    evaluate: bool | None = Field(
        default=None,
        description="Override whether evaluation metrics should be computed.",
    )
    return_column: str | None = Field(
        default=None,
        description="Optional realised-return column name for evaluation metrics.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Optionally cap the number of prediction rows returned.",
    )


class ReloadRequest(BaseModel):
    """Payload for hot-reloading the agent."""

    checkpoint_path: Optional[Path] = Field(
        default=None,
        description="Checkpoint to restore; defaults to the service setting when omitted.",
    )


class PredictionResponse(BaseModel):
    """Structured response wrapping predictions, metrics, and telemetry."""

    rows: int
    predictions: list[dict[str, Any]]
    metrics: dict[str, float | str | None] | None = None
    telemetry: dict[str, Any] | None = None


class ConfigResponse(BaseModel):
    """Serialised experiment configuration consumed by the agent."""

    config: dict[str, Any]


class ReloadResponse(BaseModel):
    """Acknowledgement returned when the agent is reloaded."""

    checkpoint_path: str | None


class CurriculumResponse(BaseModel):
    """Resolved curriculum parameters for the requested number of epochs."""

    epochs: int
    stages: list[dict[str, Any]]


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses into JSON-serialisable structures."""

    if is_dataclass(obj):
        return {name: _dataclass_to_dict(getattr(obj, name)) for name in [f.name for f in fields(obj)]}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: _dataclass_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(value) for value in obj]
    return obj


class ServiceState:
    """Mutable state shared across FastAPI endpoints."""

    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self._lock = Lock()
        self._config: ExperimentConfig | None = None
        self._agent: MarketNNPlusUltraAgent | None = None
        self.reload()

    @property
    def config(self) -> ExperimentConfig:
        if self._config is None:
            raise RuntimeError("Service not initialised")
        return self._config

    def reload(self, checkpoint_path: Optional[Path] = None) -> None:
        """Reload the experiment configuration and agent checkpoint."""

        with self._lock:
            config = load_experiment_from_file(self.settings.config_path)
            checkpoint = checkpoint_path if checkpoint_path is not None else self.settings.checkpoint_path
            agent = MarketNNPlusUltraAgent(
                experiment_config=config,
                checkpoint_path=checkpoint,
                device=self.settings.device,
            )
            self._config = config
            self._agent = agent

    def run_agent(self, *, evaluate: bool, return_column: str) -> tuple[AgentRunResult, dict[str, Any]]:
        """Execute the agent and capture telemetry for responses."""

        with self._lock:
            if self._agent is None:
                raise RuntimeError("Agent not initialised")
            result = self._agent.run(evaluate=evaluate, return_column=return_column)
            telemetry = self._build_telemetry(result)
            return result, telemetry

    def _build_telemetry(self, result: AgentRunResult) -> dict[str, Any]:
        agent = self._agent
        config = self._config
        feature_columns: list[str] = []
        horizon: int | None = None
        window_size: int | None = None
        stride: int | None = None
        symbols: list[str] = []
        if agent is not None:
            feature_columns = agent.feature_columns
        if config is not None:
            horizon = config.model.horizon
            window_size = config.data.window_size
            stride = config.data.stride
            symbols = list(config.data.symbol_universe or [])
        return {
            "prediction_rows": len(result.predictions),
            "feature_columns": feature_columns,
            "horizon": horizon,
            "window_size": window_size,
            "stride": stride,
            "symbols": symbols,
        }


def _serialise_predictions(result: AgentRunResult, *, limit: Optional[int], max_rows: Optional[int]) -> tuple[list[dict[str, Any]], int]:
    frame = result.predictions
    if max_rows is not None and max_rows > 0:
        frame = frame.head(max_rows)
    if limit is not None:
        frame = frame.head(limit)
    rows = len(frame)
    records = frame.to_dict(orient="records")
    return jsonable_encoder(records), rows


def _serialise_metrics(metrics: Optional[dict[str, float | object]]) -> Optional[dict[str, object]]:
    if metrics is None:
        return None

    serialised: dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if math.isnan(value):
                serialised[key] = None
            elif math.isinf(value):
                serialised[key] = "inf" if value > 0 else "-inf"
            else:
                serialised[key] = float(value)
        else:
            serialised[key] = value
    return serialised


def create_app(settings: ServiceSettings) -> FastAPI:
    """Build a FastAPI application bound to ``settings``."""

    state = ServiceState(settings)
    app = FastAPI(title="Market NN Plus Ultra Service", version="0.1.0")

    def get_state() -> ServiceState:
        return state

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "config_path": str(settings.config_path),
            "checkpoint_path": str(settings.checkpoint_path) if settings.checkpoint_path else None,
        }

    @app.get("/config", response_model=ConfigResponse)
    async def read_config(service_state: ServiceState = Depends(get_state)) -> ConfigResponse:
        config_dict = _dataclass_to_dict(service_state.config)
        return ConfigResponse(config=config_dict)

    @app.get("/curriculum", response_model=CurriculumResponse)
    async def curriculum(
        epochs: int = Query(
            default=None,
            gt=0,
            description="Number of epochs to resolve; defaults to trainer.max_epochs when omitted.",
        ),
        service_state: ServiceState = Depends(get_state),
    ) -> CurriculumResponse:
        config = service_state.config
        total_epochs = epochs or config.trainer.max_epochs
        stages = summarise_curriculum_profile(config.data, total_epochs)
        payload = [asdict(stage) for stage in stages]
        return CurriculumResponse(epochs=total_epochs, stages=jsonable_encoder(payload))

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        request: PredictionRequest,
        service_state: ServiceState = Depends(get_state),
    ) -> PredictionResponse:
        evaluate = request.evaluate if request.evaluate is not None else settings.evaluate_by_default
        return_column = request.return_column or settings.return_column

        try:
            result, telemetry = await run_in_threadpool(
                service_state.run_agent,
                evaluate=evaluate,
                return_column=return_column,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        predictions, rows = _serialise_predictions(
            result,
            limit=request.limit,
            max_rows=settings.max_prediction_rows,
        )
        metrics = _serialise_metrics(result.metrics if evaluate else None)
        return PredictionResponse(rows=rows, predictions=predictions, metrics=metrics, telemetry=telemetry)

    @app.post("/reload", response_model=ReloadResponse)
    async def reload(
        request: ReloadRequest,
        service_state: ServiceState = Depends(get_state),
    ) -> ReloadResponse:
        checkpoint = request.checkpoint_path
        try:
            await run_in_threadpool(service_state.reload, checkpoint_path=checkpoint)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return ReloadResponse(checkpoint_path=str(checkpoint) if checkpoint else str(settings.checkpoint_path) if settings.checkpoint_path else None)

    return app


__all__ = ["ServiceSettings", "create_app"]
