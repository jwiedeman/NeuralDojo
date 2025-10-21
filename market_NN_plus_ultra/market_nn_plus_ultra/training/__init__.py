"""Training utilities for Market NN Plus Ultra."""

from .config import (
    CurriculumConfig,
    CurriculumStage,
    DataConfig,
    DiagnosticsConfig,
    ExperimentConfig,
    GuardrailConfig,
    ModelConfig,
    OptimizerConfig,
    PretrainingConfig,
    ReplayBufferConfig,
    RiskObjectiveConfig,
    ReinforcementConfig,
    TrainerConfig,
)
from .checkpoints import load_backbone_from_checkpoint
from .train_loop import (
    MarketDataModule,
    MarketLightningModule,
    instantiate_modules,
    load_experiment_from_file,
    TrainingRunResult,
    run_training,
)
from .benchmark import (
    BenchmarkScenario,
    TrainerOverrides,
    flatten_benchmark_result,
    iter_scenarios,
    prepare_config_for_scenario,
)
from .curriculum import (
    CurriculumCallback,
    CurriculumParameters,
    CurriculumScheduler,
    summarise_curriculum_profile,
)
from .diagnostics import (
    DiagnosticsThresholds,
    RunningMoments,
    TrainingDiagnosticsCallback,
)
from .pretrain_loop import (
    ContrastiveTimeSeriesLightningModule,
    MaskedTimeSeriesLightningModule,
    instantiate_pretraining_module,
    run_pretraining,
)
from .reinforcement import (
    MarketPolicyNetwork,
    ReinforcementRunResult,
    ReinforcementUpdate,
    RolloutTelemetry,
    run_reinforcement_finetuning,
)

__all__ = [
    "DataConfig",
    "DiagnosticsConfig",
    "ModelConfig",
    "OptimizerConfig",
    "TrainerConfig",
    "ExperimentConfig",
    "GuardrailConfig",
    "PretrainingConfig",
    "ReinforcementConfig",
    "RiskObjectiveConfig",
    "ReplayBufferConfig",
    "CurriculumConfig",
    "CurriculumStage",
    "CurriculumScheduler",
    "CurriculumParameters",
    "CurriculumCallback",
    "summarise_curriculum_profile",
    "TrainingDiagnosticsCallback",
    "DiagnosticsThresholds",
    "RunningMoments",
    "MarketLightningModule",
    "MarketDataModule",
    "instantiate_modules",
    "load_experiment_from_file",
    "TrainingRunResult",
    "run_training",
    "load_backbone_from_checkpoint",
    "BenchmarkScenario",
    "TrainerOverrides",
    "prepare_config_for_scenario",
    "iter_scenarios",
    "flatten_benchmark_result",
    "MaskedTimeSeriesLightningModule",
    "ContrastiveTimeSeriesLightningModule",
    "instantiate_pretraining_module",
    "run_pretraining",
    "MarketPolicyNetwork",
    "run_reinforcement_finetuning",
    "ReinforcementRunResult",
    "ReinforcementUpdate",
    "RolloutTelemetry",
]
