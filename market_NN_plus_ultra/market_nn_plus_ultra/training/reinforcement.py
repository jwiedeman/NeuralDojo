"""Policy-gradient fine-tuning for the Market NN Plus Ultra agent."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
import math
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
import torch.multiprocessing as mp

from ..trading.pnl import TradingCosts, differentiable_pnl, price_to_returns
from ..trading.risk import compute_risk_metrics
from .config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    ReplayBufferConfig,
    ReinforcementConfig,
    RiskObjectiveConfig,
    TrainerConfig,
)
from .checkpoints import load_backbone_from_checkpoint
from .curriculum import CurriculumParameters
from .train_loop import MarketDataModule, MarketLightningModule


def _build_backbone(model_config: ModelConfig) -> nn.Module:
    """Instantiate the backbone described by ``model_config``."""

    module = MarketLightningModule(model_config, OptimizerConfig())
    return module.backbone


class MarketPolicyNetwork(nn.Module):
    """Gaussian policy/value network built on top of the trading backbone."""

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        self.model_config = model_config
        self.backbone = _build_backbone(model_config)
        self.max_horizon = model_config.horizon
        self.horizon = model_config.horizon
        self.action_dim = model_config.output_dim
        self.policy_head = nn.Sequential(
            nn.LayerNorm(model_config.model_dim),
            nn.Linear(model_config.model_dim, model_config.model_dim),
            nn.GELU(),
            nn.Linear(model_config.model_dim, self.max_horizon * self.action_dim),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(model_config.model_dim),
            nn.Linear(model_config.model_dim, model_config.model_dim),
            nn.GELU(),
            nn.Linear(model_config.model_dim, self.max_horizon),
        )
        self.log_std = nn.Parameter(torch.zeros(self.max_horizon, self.action_dim))

    def set_horizon(self, horizon: int) -> None:
        if horizon < 1:
            raise ValueError("Horizon must be positive")
        if horizon > self.max_horizon:
            raise ValueError(
                f"Horizon {horizon} exceeds maximum configured horizon {self.max_horizon}"
            )
        self.horizon = horizon

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        last_state = hidden[:, -1, :]
        mean = self.policy_head(last_state)
        mean = mean.view(features.size(0), self.max_horizon, self.action_dim)
        mean = mean[:, : self.horizon, :]
        value = self.value_head(last_state)
        value = value[:, : self.horizon]
        return mean, value

    def _distribution(self, mean: torch.Tensor) -> Normal:
        std = torch.exp(self.log_std[: self.horizon]).clamp(min=1e-4)
        std = std.unsqueeze(0).expand_as(mean)
        return Normal(mean, std)

    def act(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(features)
        dist = self._distribution(mean)
        actions = dist.rsample()
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return actions, log_prob, entropy, value

    def evaluate_actions(self, features: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(features)
        if actions.size(1) != self.horizon:
            raise ValueError(
                "Action horizon does not match policy horizon: "
                f"expected {self.horizon}, got {actions.size(1)}"
            )
        dist = self._distribution(mean)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


@dataclass(slots=True)
class RolloutBatch:
    """Rollout buffer used by PPO updates."""

    features: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    rewards: torch.Tensor

    def to(self, device: torch.device | str) -> "RolloutBatch":
        device = torch.device(device)
        return RolloutBatch(
            features=self.features.to(device),
            actions=self.actions.to(device),
            log_probs=self.log_probs.to(device),
            values=self.values.to(device),
            advantages=self.advantages.to(device),
            returns=self.returns.to(device),
            rewards=self.rewards.to(device),
        )

    def detach(self) -> "RolloutBatch":
        return RolloutBatch(
            features=self.features.detach(),
            actions=self.actions.detach(),
            log_probs=self.log_probs.detach(),
            values=self.values.detach(),
            advantages=self.advantages.detach(),
            returns=self.returns.detach(),
            rewards=self.rewards.detach(),
        )

    def __len__(self) -> int:
        return int(self.features.size(0))

    def select(self, indices: torch.Tensor) -> "RolloutBatch":
        return RolloutBatch(
            features=self.features.index_select(0, indices),
            actions=self.actions.index_select(0, indices),
            log_probs=self.log_probs.index_select(0, indices),
            values=self.values.index_select(0, indices),
            advantages=self.advantages.index_select(0, indices),
            returns=self.returns.index_select(0, indices),
            rewards=self.rewards.index_select(0, indices),
        )

    def slice(self, start: int, end: int | None = None) -> "RolloutBatch":
        return RolloutBatch(
            features=self.features[start:end],
            actions=self.actions[start:end],
            log_probs=self.log_probs[start:end],
            values=self.values[start:end],
            advantages=self.advantages[start:end],
            returns=self.returns[start:end],
        rewards=self.rewards[start:end],
    )


def _concat_rollout_batches(batches: Sequence[RolloutBatch]) -> RolloutBatch:
    valid = [batch for batch in batches if batch is not None]
    if not valid:
        raise ValueError("At least one rollout batch must be provided")
    return RolloutBatch(
        features=torch.cat([batch.features for batch in valid], dim=0),
        actions=torch.cat([batch.actions for batch in valid], dim=0),
        log_probs=torch.cat([batch.log_probs for batch in valid], dim=0),
        values=torch.cat([batch.values for batch in valid], dim=0),
        advantages=torch.cat([batch.advantages for batch in valid], dim=0),
        returns=torch.cat([batch.returns for batch in valid], dim=0),
        rewards=torch.cat([batch.rewards for batch in valid], dim=0),
    )


@dataclass(slots=True)
class RolloutTelemetry:
    """Lightweight telemetry captured alongside PPO rollouts."""

    batches: int = 0
    sequences: int = 0
    steps: int = 0
    reward_sum: float = 0.0
    reward_sq_sum: float = 0.0
    duration: float = 0.0

    def merge(self, other: "RolloutTelemetry") -> "RolloutTelemetry":
        self.batches += other.batches
        self.sequences += other.sequences
        self.steps += other.steps
        self.reward_sum += other.reward_sum
        self.reward_sq_sum += other.reward_sq_sum
        self.duration += other.duration
        return self

    def mean_reward(self) -> float:
        if self.steps == 0:
            return 0.0
        return self.reward_sum / self.steps

    def reward_std(self) -> float:
        if self.steps == 0:
            return 0.0
        mean = self.mean_reward()
        variance = max((self.reward_sq_sum / self.steps) - mean**2, 0.0)
        return math.sqrt(variance)

    def sequences_per_second(self) -> float:
        if self.duration <= 0.0:
            return 0.0
        return self.sequences / self.duration

    def steps_per_second(self) -> float:
        if self.duration <= 0.0:
            return 0.0
        return self.steps / self.duration


def _merge_rollout_telemetry(telemetries: Sequence[RolloutTelemetry]) -> RolloutTelemetry:
    telemetry = RolloutTelemetry()
    for item in telemetries:
        telemetry.merge(item)
    return telemetry


class RolloutReplayBuffer:
    """Fixed-capacity buffer that stores PPO rollouts on CPU."""

    def __init__(self, config: ReplayBufferConfig) -> None:
        self.config = config
        self._buffer: RolloutBatch | None = None

    def add(self, batch: RolloutBatch) -> None:
        cpu_batch = batch.detach().to(torch.device("cpu"))
        if self._buffer is None:
            self._buffer = cpu_batch
        else:
            self._buffer = _concat_rollout_batches([self._buffer, cpu_batch])
        excess = len(self._buffer) - self.config.capacity
        if excess > 0:
            self._buffer = self._buffer.slice(excess)

    def can_sample(self) -> bool:
        return self._buffer is not None and len(self._buffer) >= self.config.min_samples

    def sample(self, count: int) -> RolloutBatch | None:
        if self._buffer is None or not self.can_sample():
            return None
        total = len(self._buffer)
        count = max(0, min(count, total))
        if count == 0:
            return None
        indices = torch.randperm(total)[:count]
        return self._buffer.select(indices)


def _infinite_iterator(loader: Iterable[dict[str, torch.Tensor]]) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def _normalise_trainer_for_worker(config: TrainerConfig) -> TrainerConfig:
    trainer = replace(config)
    trainer.accelerator = "cpu"
    trainer.devices = None
    trainer.num_workers = 0
    trainer.persistent_workers = False
    return trainer


def _rollout_worker_loop(
    worker_id: int,
    data_config: DataConfig,
    trainer_config: TrainerConfig,
    model_config: ModelConfig,
    reinforcement_config: ReinforcementConfig,
    base_seed: int,
    device: str,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(base_seed + worker_id)

    try:
        worker_device = torch.device(device)
        reinforcement = replace(reinforcement_config)
        reinforcement.rollout_workers = 1
        trainer = _normalise_trainer_for_worker(trainer_config)
        data_module = MarketDataModule(data_config, trainer, seed=base_seed + worker_id)
        data_module.setup("fit")
        train_loader = data_module.train_dataloader()
        if len(train_loader) == 0:
            raise RuntimeError(
                "Training dataloader is empty in rollout worker; check dataset configuration."
            )
        iterator = _infinite_iterator(train_loader)
        policy = MarketPolicyNetwork(model_config).to(worker_device)
        initial_curriculum: CurriculumParameters | None = getattr(
            data_module, "current_curriculum", None
        )
        if initial_curriculum is not None:
            policy.set_horizon(initial_curriculum.horizon)

        response_queue.put(("ready", None))

        while True:
            command, payload = request_queue.get()
            if command == "stop":
                break
            if command != "collect":
                continue

            state_dict, step = payload
            if data_module.has_curriculum:
                changed = data_module.step_curriculum(step)
                if changed is not None:
                    train_loader = data_module.train_dataloader()
                    if len(train_loader) == 0:
                        raise RuntimeError(
                            "Training dataloader is empty after curriculum step; adjust schedule or dataset length."
                        )
                    iterator = _infinite_iterator(train_loader)
                current = getattr(data_module, "current_curriculum", None)
                if current is not None:
                    policy.set_horizon(current.horizon)
            policy.load_state_dict({key: tensor.to(worker_device) for key, tensor in state_dict.items()})
            rollout, telemetry = _collect_rollout(
                policy, iterator, reinforcement, worker_device
            )
            response_queue.put(
                (
                    "result",
                    (
                        rollout.detach().to(torch.device("cpu")),
                        telemetry,
                    ),
                )
            )
    except Exception as exc:  # pragma: no cover - surfaced via response queue
        response_queue.put(("error", repr(exc)))


class ParallelRolloutManager:
    """Manage a pool of worker processes that collect PPO rollouts."""

    def __init__(self, experiment: ExperimentConfig, reinforcement: ReinforcementConfig) -> None:
        if reinforcement.rollout_workers < 1:
            raise ValueError("reinforcement.rollout_workers must be at least 1")

        self._ctx = mp.get_context("spawn")
        self._processes: list[mp.Process] = []
        self._requests: list[mp.Queue] = []
        self._responses: list[mp.Queue] = []
        self._closed = False
        self._model_config = experiment.model
        self._data_config = experiment.data
        self._trainer_config = experiment.trainer
        self._reinforcement = reinforcement
        self._seed = experiment.seed

        for worker_id in range(reinforcement.rollout_workers):
            request = self._ctx.Queue()
            response = self._ctx.Queue()
            process = self._ctx.Process(
                target=_rollout_worker_loop,
                args=(
                    worker_id,
                    self._data_config,
                    self._trainer_config,
                    self._model_config,
                    self._reinforcement,
                    self._seed,
                    reinforcement.worker_device,
                    request,
                    response,
                ),
            )
            process.daemon = True
            process.start()

            kind, payload = response.get()
            if kind == "error":
                process.join(timeout=1.0)
                self.close()
                raise RuntimeError(f"Rollout worker failed to start: {payload}")
            if kind != "ready":
                self.close()
                raise RuntimeError(f"Unexpected worker initialisation message: {kind}")

            self._processes.append(process)
            self._requests.append(request)
            self._responses.append(response)

    def collect(self, policy: MarketPolicyNetwork, step: int) -> Tuple[RolloutBatch, RolloutTelemetry]:
        state_dict = {
            key: tensor.detach().cpu()
            for key, tensor in policy.state_dict().items()
        }

        for queue in self._requests:
            queue.put(("collect", (state_dict, step)))

        batches: list[RolloutBatch] = []
        telemetries: list[RolloutTelemetry] = []
        for response in self._responses:
            kind, payload = response.get()
            if kind == "result":
                batch, telemetry = payload
                batches.append(batch)
                telemetries.append(telemetry)
            elif kind == "error":
                self.close()
                raise RuntimeError(f"Rollout worker error: {payload}")
            else:
                self.close()
                raise RuntimeError(f"Unexpected worker message: {kind}")

        combined_batch = _concat_rollout_batches(batches)
        combined_telemetry = _merge_rollout_telemetry(telemetries)
        return combined_batch, combined_telemetry

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        for queue in self._requests:
            with contextlib.suppress(Exception):
                queue.put(("stop", None))

        for process in self._processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

        for queue in self._responses:
            with contextlib.suppress(Exception):
                while not queue.empty():
                    queue.get_nowait()

@dataclass(slots=True)
class ReinforcementUpdate:
    """Summary of a single PPO update."""

    update: int
    mean_reward: float
    reward_std: float
    policy_loss: float
    value_loss: float
    entropy: float
    samples: int
    collection_time: float
    samples_per_second: float
    steps_per_second: float
    curriculum: CurriculumParameters | None = None


@dataclass(slots=True)
class ReinforcementRunResult:
    """Container holding PPO run statistics and the trained policy state."""

    updates: List[ReinforcementUpdate]
    policy_state_dict: dict[str, torch.Tensor]
    evaluation_metrics: Dict[str, float]


def _apply_risk_objectives(pnl: torch.Tensor, config: RiskObjectiveConfig) -> torch.Tensor:
    """Blend risk metrics into the per-step reward signal when configured."""

    if pnl.ndim < 2:
        raise ValueError("PnL tensor must have shape [batch, horizon] or higher")

    if not config.is_active():
        return pnl

    metrics = compute_risk_metrics(pnl, alpha=config.cvar_alpha, dim=-1)
    horizon = pnl.size(-1)

    bonus = torch.zeros_like(metrics.sharpe)
    if config.sharpe_weight != 0.0:
        bonus = bonus + config.sharpe_weight * metrics.sharpe
    if config.sortino_weight != 0.0:
        bonus = bonus + config.sortino_weight * metrics.sortino
    if config.drawdown_weight != 0.0:
        bonus = bonus - config.drawdown_weight * metrics.drawdown
    if config.cvar_weight != 0.0:
        cvar_penalty = torch.clamp(-metrics.cvar, min=0.0)
        bonus = bonus - config.cvar_weight * cvar_penalty

    per_step_adjustment = (config.reward_scale * bonus).unsqueeze(-1) / float(horizon)
    return pnl + per_step_adjustment


def _compute_gae(
    rewards: torch.Tensor, values: torch.Tensor, gamma: float, gae_lambda: float
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, horizon = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)
    last_value = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(horizon)):
        delta = rewards[:, t] + gamma * last_value - values[:, t]
        last_gae = delta + gamma * gae_lambda * last_gae
        advantages[:, t] = last_gae
        last_value = values[:, t]
    returns = advantages + values
    return advantages, returns


def _collect_rollout(
    policy: MarketPolicyNetwork,
    data_iterator: Iterable[dict[str, torch.Tensor]],
    reinforcement: ReinforcementConfig,
    device: torch.device,
) -> Tuple[RolloutBatch, RolloutTelemetry]:
    policy.eval()
    features_buf: List[torch.Tensor] = []
    actions_buf: List[torch.Tensor] = []
    log_prob_buf: List[torch.Tensor] = []
    value_buf: List[torch.Tensor] = []
    reward_buf: List[torch.Tensor] = []

    total_samples = 0
    costs = reinforcement.costs or TradingCosts()
    batches = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        while total_samples < reinforcement.steps_per_rollout:
            batch = next(data_iterator)
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            reference = batch.get("reference")
            if reference is not None:
                reference = reference.to(device)

            actions, log_prob, entropy, values = policy.act(features)

            batches += 1

            if reinforcement.targets_are_returns:
                future_returns = targets
            else:
                future_returns = price_to_returns(targets, reference)

            pnl = differentiable_pnl(
                actions,
                future_returns,
                costs=costs,
                activation=reinforcement.activation,
            )

            features_buf.append(features)
            actions_buf.append(actions)
            log_prob_buf.append(log_prob)
            value_buf.append(values)
            adjusted_pnl = _apply_risk_objectives(pnl, reinforcement.risk_objective)

            reward_buf.append(adjusted_pnl)
            total_samples += features.size(0)

    features_tensor = torch.cat(features_buf, dim=0)
    actions_tensor = torch.cat(actions_buf, dim=0)
    log_probs_tensor = torch.cat(log_prob_buf, dim=0)
    values_tensor = torch.cat(value_buf, dim=0)
    rewards_tensor = torch.cat(reward_buf, dim=0)

    advantages, returns = _compute_gae(
        rewards_tensor, values_tensor, reinforcement.gamma, reinforcement.gae_lambda
    )

    # Normalise advantages for numerical stability.
    adv_mean = advantages.mean()
    adv_std = advantages.std(unbiased=False).clamp(min=1e-6)
    advantages = (advantages - adv_mean) / adv_std

    telemetry = RolloutTelemetry(
        batches=batches,
        sequences=int(features_tensor.size(0)),
        steps=int(rewards_tensor.numel()),
        reward_sum=float(rewards_tensor.sum().item()),
        reward_sq_sum=float(rewards_tensor.square().sum().item()),
        duration=time.perf_counter() - start_time,
    )

    return (
        RolloutBatch(
            features=features_tensor,
            actions=actions_tensor,
            log_probs=log_probs_tensor,
            values=values_tensor,
            advantages=advantages,
            returns=returns,
            rewards=rewards_tensor,
        ),
        telemetry,
    )


def _evaluate_policy(
    policy: MarketPolicyNetwork,
    dataloader: Iterable[dict[str, torch.Tensor]],
    reinforcement: ReinforcementConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Run the policy in evaluation mode and aggregate ROI-style diagnostics."""

    policy.eval()
    costs = reinforcement.costs or TradingCosts()
    pnl_series: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            reference = batch.get("reference")
            if reference is not None:
                reference = reference.to(device)

            mean, _ = policy.forward(features)
            if reinforcement.targets_are_returns:
                future_returns = targets
            else:
                future_returns = price_to_returns(targets, reference)

            pnl = differentiable_pnl(
                mean,
                future_returns,
                costs=costs,
                activation=reinforcement.activation,
            )
            pnl_series.append(pnl)

    if not pnl_series:
        return {}

    pnl_tensor = torch.cat(pnl_series, dim=0)
    risk = compute_risk_metrics(pnl_tensor, dim=-1)
    roi = pnl_tensor.sum(dim=-1)
    summary: Dict[str, float] = {
        "roi_mean": float(roi.mean().item()),
        "roi_std": float(roi.std(unbiased=False).item()),
        "sharpe": float(risk.sharpe.mean().item()),
        "sortino": float(risk.sortino.mean().item()),
        "max_drawdown": float(risk.drawdown.mean().item()),
        "cvar": float(risk.cvar.mean().item()),
    }
    return summary


def _ppo_update(
    policy: MarketPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    reinforcement: ReinforcementConfig,
    *,
    telemetry: RolloutTelemetry | None = None,
) -> ReinforcementUpdate:
    policy.train()

    num_samples = len(rollout)
    minibatch = max(1, min(reinforcement.minibatch_size, num_samples))
    indices = torch.arange(num_samples, device=rollout.features.device)

    policy_loss_total = 0.0
    value_loss_total = 0.0
    entropy_total = 0.0
    updates = 0

    for _ in range(reinforcement.policy_epochs):
        perm = indices[torch.randperm(num_samples)]
        for start in range(0, num_samples, minibatch):
            batch_idx = perm[start : start + minibatch]
            features = rollout.features[batch_idx]
            actions = rollout.actions[batch_idx]
            old_log_prob = rollout.log_probs[batch_idx]
            advantages = rollout.advantages[batch_idx]
            returns = rollout.returns[batch_idx]

            log_prob, entropy, values = policy.evaluate_actions(features, actions)
            ratio = torch.exp(log_prob - old_log_prob)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - reinforcement.clip_ratio, 1.0 + reinforcement.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns)
            entropy_loss = entropy.mean()

            loss = policy_loss + reinforcement.value_coef * value_loss - reinforcement.entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), reinforcement.max_grad_norm)
            optimizer.step()

            policy_loss_total += policy_loss.item()
            value_loss_total += value_loss.item()
            entropy_total += entropy_loss.item()
            updates += 1

    mean_reward = rollout.rewards.mean().item()
    reward_std = rollout.rewards.std(unbiased=False).item()
    collection_time = telemetry.duration if telemetry is not None else 0.0
    samples_per_second = telemetry.sequences_per_second() if telemetry is not None else 0.0
    steps_per_second = telemetry.steps_per_second() if telemetry is not None else 0.0
    denom = max(1, updates)
    return ReinforcementUpdate(
        update=0,
        mean_reward=mean_reward,
        reward_std=reward_std,
        policy_loss=policy_loss_total / denom,
        value_loss=value_loss_total / denom,
        entropy=entropy_total / denom,
        samples=num_samples,
        collection_time=collection_time,
        samples_per_second=samples_per_second,
        steps_per_second=steps_per_second,
    )


def run_reinforcement_finetuning(
    experiment_config: ExperimentConfig,
    reinforcement_config: Optional[ReinforcementConfig] = None,
    *,
    checkpoint_path: Optional[str | Path] = None,
    pretrain_checkpoint_path: Optional[str | Path] = None,
    device: str | torch.device = "cpu",
) -> ReinforcementRunResult:
    """Run PPO fine-tuning on top of the supervised market model."""

    if reinforcement_config is not None:
        reinforcement = reinforcement_config
    elif experiment_config.reinforcement is not None:
        reinforcement = replace(experiment_config.reinforcement)
    else:
        reinforcement = ReinforcementConfig()
    torch.manual_seed(experiment_config.seed)
    device = torch.device(device)

    if checkpoint_path is not None and pretrain_checkpoint_path is not None:
        raise ValueError("Specify only one of 'checkpoint_path' or 'pretrain_checkpoint_path'")

    policy = MarketPolicyNetwork(experiment_config.model).to(device)

    if checkpoint_path is not None:
        load_backbone_from_checkpoint(policy.backbone, checkpoint_path, device=device)
    elif pretrain_checkpoint_path is not None:
        load_backbone_from_checkpoint(
            policy.backbone,
            pretrain_checkpoint_path,
            device=device,
        )

    optimizer = torch.optim.Adam(policy.parameters(), lr=reinforcement.learning_rate)

    data_module = MarketDataModule(experiment_config.data, experiment_config.trainer, seed=experiment_config.seed)
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    if len(train_loader) == 0:
        raise RuntimeError("Training dataloader is empty; ensure the dataset yields at least one batch")

    initial_curriculum: CurriculumParameters | None = getattr(data_module, "current_curriculum", None)
    if initial_curriculum is not None:
        policy.set_horizon(initial_curriculum.horizon)

    iterator = _infinite_iterator(train_loader)
    updates: List[ReinforcementUpdate] = []

    buffer: RolloutReplayBuffer | None = None
    if reinforcement.replay_buffer.enabled:
        if reinforcement.replay_buffer.capacity <= 0:
            raise ValueError("replay_buffer.capacity must be positive when enabled")
        if reinforcement.replay_buffer.min_samples <= 0:
            raise ValueError("replay_buffer.min_samples must be positive when enabled")
        buffer = RolloutReplayBuffer(reinforcement.replay_buffer)

    sample_ratio = (
        reinforcement.replay_buffer.sample_ratio if reinforcement.replay_buffer.enabled else 0.0
    )
    if sample_ratio < 0.0:
        raise ValueError("replay_buffer.sample_ratio must be non-negative")

    manager: ParallelRolloutManager | None = None
    if reinforcement.rollout_workers < 1:
        raise ValueError("reinforcement.rollout_workers must be at least 1")

    evaluation_metrics: Dict[str, float] = {}
    try:
        if reinforcement.rollout_workers > 1:
            manager = ParallelRolloutManager(experiment_config, reinforcement)

        for step in range(reinforcement.total_updates):
            if data_module.has_curriculum:
                changed = data_module.step_curriculum(step)
                if changed is not None:
                    train_dataset = getattr(data_module, "train_dataset", None)
                    if train_dataset is None or len(train_dataset) == 0:  # type: ignore[arg-type]
                        raise RuntimeError(
                            "Curriculum stage produced an empty training dataset; adjust schedule or dataset length."
                        )
                    if manager is None:
                        train_loader = data_module.train_dataloader()
                        if len(train_loader) == 0:
                            raise RuntimeError(
                                "Training dataloader is empty after curriculum step; adjust schedule or dataset length."
                            )
                        iterator = _infinite_iterator(train_loader)

            current_curriculum: CurriculumParameters | None = getattr(
                data_module, "current_curriculum", None
            )
            if (
                current_curriculum is not None
                and policy.horizon != current_curriculum.horizon
            ):
                policy.set_horizon(current_curriculum.horizon)

            if manager is not None:
                rollout, telemetry = manager.collect(policy, step)
            else:
                rollout, telemetry = _collect_rollout(policy, iterator, reinforcement, device)

            rollout = rollout.to(device)
            combined = rollout
            combined_telemetry = telemetry
            if buffer is not None:
                ratio = min(sample_ratio, 1.0)
                replay_batch: RolloutBatch | None = None
                if ratio > 0.0 and buffer.can_sample():
                    requested = max(1, int(round(len(rollout) * ratio)))
                    replay_batch = buffer.sample(requested)
                buffer.add(rollout)
                if replay_batch is not None:
                    combined = _concat_rollout_batches([rollout, replay_batch.to(device)])
                    combined_telemetry = telemetry

            update_stats = _ppo_update(
                policy,
                optimizer,
                combined,
                reinforcement,
                telemetry=combined_telemetry,
            )
            updates.append(
                ReinforcementUpdate(
                    update=step,
                    mean_reward=update_stats.mean_reward,
                    reward_std=update_stats.reward_std,
                    policy_loss=update_stats.policy_loss,
                    value_loss=update_stats.value_loss,
                    entropy=update_stats.entropy,
                    samples=update_stats.samples,
                    collection_time=update_stats.collection_time,
                    samples_per_second=update_stats.samples_per_second,
                    steps_per_second=update_stats.steps_per_second,
                    curriculum=current_curriculum,
                )
            )
    finally:
        if manager is not None:
            manager.close()

    with torch.no_grad():
        try:
            eval_loader = data_module.val_dataloader()
        except Exception:  # pragma: no cover - defensive guard for unexpected datamodule issues
            eval_loader = None
        if eval_loader is not None:
            evaluation_metrics = _evaluate_policy(policy, eval_loader, reinforcement, device)

    state_dict = {key: tensor.detach().cpu() for key, tensor in policy.state_dict().items()}
    return ReinforcementRunResult(
        updates=updates,
        policy_state_dict=state_dict,
        evaluation_metrics=evaluation_metrics,
    )


__all__ = [
    "MarketPolicyNetwork",
    "ReinforcementRunResult",
    "ReinforcementUpdate",
    "run_reinforcement_finetuning",
]
