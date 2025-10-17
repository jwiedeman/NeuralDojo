"""Training utilities for Market NN Plus Ultra."""

from .config import (
    CurriculumConfig,
    CurriculumStage,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    PretrainingConfig,
    ReinforcementConfig,
    TrainerConfig,
)
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
    run_reinforcement_finetuning,
)

__all__ = [
    "DataConfig",
    "ModelConfig",
    "OptimizerConfig",
    "TrainerConfig",
    "ExperimentConfig",
    "PretrainingConfig",
    "ReinforcementConfig",
    "CurriculumConfig",
    "CurriculumStage",
    "CurriculumScheduler",
    "CurriculumParameters",
    "CurriculumCallback",
    "MarketLightningModule",
    "MarketDataModule",
    "instantiate_modules",
    "load_experiment_from_file",
    "TrainingRunResult",
    "run_training",
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
]
