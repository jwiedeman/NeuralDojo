"""Policy-gradient fine-tuning for the Market NN Plus Ultra agent."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from ..trading.pnl import TradingCosts, differentiable_pnl, price_to_returns
from .config import ExperimentConfig, ModelConfig, OptimizerConfig, ReinforcementConfig
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
        self.horizon = model_config.horizon
        self.action_dim = model_config.output_dim
        self.policy_head = nn.Sequential(
            nn.LayerNorm(model_config.model_dim),
            nn.Linear(model_config.model_dim, model_config.model_dim),
            nn.GELU(),
            nn.Linear(model_config.model_dim, self.horizon * self.action_dim),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(model_config.model_dim),
            nn.Linear(model_config.model_dim, model_config.model_dim),
            nn.GELU(),
            nn.Linear(model_config.model_dim, self.horizon),
        )
        self.log_std = nn.Parameter(torch.zeros(self.horizon, self.action_dim))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        last_state = hidden[:, -1, :]
        mean = self.policy_head(last_state)
        mean = mean.view(features.size(0), self.horizon, self.action_dim)
        value = self.value_head(last_state)
        return mean, value

    def _distribution(self, mean: torch.Tensor) -> Normal:
        std = torch.exp(self.log_std).clamp(min=1e-4)
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


@dataclass(slots=True)
class ReinforcementUpdate:
    """Summary of a single PPO update."""

    update: int
    mean_reward: float
    policy_loss: float
    value_loss: float
    entropy: float


@dataclass(slots=True)
class ReinforcementRunResult:
    """Container holding PPO run statistics and the trained policy state."""

    updates: List[ReinforcementUpdate]
    policy_state_dict: dict[str, torch.Tensor]


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
) -> RolloutBatch:
    policy.eval()
    features_buf: List[torch.Tensor] = []
    actions_buf: List[torch.Tensor] = []
    log_prob_buf: List[torch.Tensor] = []
    value_buf: List[torch.Tensor] = []
    reward_buf: List[torch.Tensor] = []

    total_samples = 0
    costs = reinforcement.costs or TradingCosts()

    with torch.no_grad():
        while total_samples < reinforcement.steps_per_rollout:
            batch = next(data_iterator)
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            reference = batch.get("reference")
            if reference is not None:
                reference = reference.to(device)

            actions, log_prob, entropy, values = policy.act(features)

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
            reward_buf.append(pnl)
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

    return RolloutBatch(
        features=features_tensor,
        actions=actions_tensor,
        log_probs=log_probs_tensor,
        values=values_tensor,
        advantages=advantages,
        returns=returns,
        rewards=rewards_tensor,
    )


def _ppo_update(
    policy: MarketPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    reinforcement: ReinforcementConfig,
) -> ReinforcementUpdate:
    policy.train()

    num_samples = rollout.features.size(0)
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
    denom = max(1, updates)
    return ReinforcementUpdate(
        update=0,
        mean_reward=mean_reward,
        policy_loss=policy_loss_total / denom,
        value_loss=value_loss_total / denom,
        entropy=entropy_total / denom,
    )


def run_reinforcement_finetuning(
    experiment_config: ExperimentConfig,
    reinforcement_config: Optional[ReinforcementConfig] = None,
    *,
    checkpoint_path: Optional[str | Path] = None,
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

    policy = MarketPolicyNetwork(experiment_config.model).to(device)

    if checkpoint_path is not None:
        module = MarketLightningModule.from_checkpoint(
            checkpoint_path,
            model_config=experiment_config.model,
            optimizer_config=experiment_config.optimizer,
            map_location=device,
        )
        policy.backbone.load_state_dict(module.backbone.state_dict())

    optimizer = torch.optim.Adam(policy.parameters(), lr=reinforcement.learning_rate)

    data_module = MarketDataModule(experiment_config.data, experiment_config.trainer, seed=experiment_config.seed)
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    if len(train_loader) == 0:
        raise RuntimeError("Training dataloader is empty; ensure the dataset yields at least one batch")

    def infinite_iterator(loader: Iterable[dict[str, torch.Tensor]]):
        while True:
            for batch in loader:
                yield batch

    iterator = infinite_iterator(train_loader)
    updates: List[ReinforcementUpdate] = []

    for step in range(reinforcement.total_updates):
        rollout = _collect_rollout(policy, iterator, reinforcement, device)
        update_stats = _ppo_update(policy, optimizer, rollout, reinforcement)
        updates.append(
            ReinforcementUpdate(
                update=step,
                mean_reward=update_stats.mean_reward,
                policy_loss=update_stats.policy_loss,
                value_loss=update_stats.value_loss,
                entropy=update_stats.entropy,
            )
        )

    state_dict = {key: tensor.detach().cpu() for key, tensor in policy.state_dict().items()}
    return ReinforcementRunResult(updates=updates, policy_state_dict=state_dict)


__all__ = [
    "MarketPolicyNetwork",
    "ReinforcementRunResult",
    "ReinforcementUpdate",
    "run_reinforcement_finetuning",
]
