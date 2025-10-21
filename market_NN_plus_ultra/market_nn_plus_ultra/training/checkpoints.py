"""Checkpoint utilities shared across training entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn


def load_backbone_from_checkpoint(
    backbone: nn.Module,
    checkpoint_path: str | Path,
    *,
    device: torch.device | str | None = "cpu",
) -> None:
    """Load the backbone weights from a Lightning checkpoint.

    The helper understands checkpoints produced by both the supervised
    ``MarketLightningModule`` and the self-supervised pretraining modules.
    Only parameters that belong to the backbone are restored, ensuring
    downstream heads are freshly initialised for every consumer (training,
    reinforcement, or evaluation).
    """

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint '{path}' does not exist")

    map_location = device
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unrecognised checkpoint format for '{path}'")

    target_keys = set(backbone.state_dict().keys())
    if not target_keys:
        raise ValueError("Backbone has no parameters to load")

    prefixes: Iterable[str] = ("backbone.", "model.backbone.", "")
    for prefix in prefixes:
        matched: dict[str, torch.Tensor] = {}
        prefix_len = len(prefix)
        for key, tensor in state_dict.items():
            if prefix and not key.startswith(prefix):
                continue
            trimmed = key[prefix_len:]
            if trimmed in target_keys:
                matched[trimmed] = tensor
        if matched.keys() >= target_keys:
            backbone.load_state_dict({key: matched[key] for key in target_keys})
            return

    raise ValueError(
        "Checkpoint '%s' does not contain a compatible backbone state" % path
    )


__all__ = ["load_backbone_from_checkpoint"]

