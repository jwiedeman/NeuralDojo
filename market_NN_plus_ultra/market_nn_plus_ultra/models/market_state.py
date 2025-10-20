"""Market-state embedding helpers for Market NN Plus Ultra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn


@dataclass(slots=True)
class MarketStateFeature:
    """Metadata describing a single categorical market-state signal."""

    name: str
    token_column: str
    categories: tuple[str, ...]
    index: int


@dataclass(slots=True)
class MarketStateMetadata:
    """Collection of categorical market-state signals."""

    features: tuple[MarketStateFeature, ...]

    def feature_count(self) -> int:
        return len(self.features)

    def vocab_sizes(self) -> tuple[int, ...]:
        return tuple(len(feature.categories) for feature in self.features)

    def indices(self) -> tuple[int, ...]:
        return tuple(feature.index for feature in self.features)

    def select(self, include: Sequence[str] | None) -> "MarketStateMetadata":
        if not include:
            return self
        include_set = {name.lower() for name in include}
        selected = [
            feature
            for feature in self.features
            if feature.name.lower() in include_set
        ]
        return MarketStateMetadata(features=tuple(selected))

    @classmethod
    def from_columns(
        cls,
        columns: Iterable[tuple[str, str, Sequence[str]]],
    ) -> "MarketStateMetadata":
        features: list[MarketStateFeature] = []
        for idx, (name, token_column, categories) in enumerate(columns):
            features.append(
                MarketStateFeature(
                    name=name,
                    token_column=token_column,
                    categories=tuple(str(cat) for cat in categories),
                    index=idx,
                )
            )
        return cls(features=tuple(features))


class MarketStateEmbedding(nn.Module):
    """Lookup embeddings for categorical market-state signals."""

    def __init__(
        self,
        metadata: MarketStateMetadata,
        *,
        embedding_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if metadata.feature_count() == 0:
            raise ValueError("MarketStateEmbedding requires at least one feature")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.metadata = metadata
        self.embedding_dim = int(embedding_dim)
        self.output_dim = self.embedding_dim * metadata.feature_count()
        tables = {}
        for feature in metadata.features:
            tables[feature.name] = nn.Embedding(len(feature.categories), self.embedding_dim)
        self.tables = nn.ModuleDict(tables)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                "market-state tokens must be a 3D tensor with shape (batch, window, features)"
            )
        if tokens.shape[-1] != self.metadata.feature_count():
            raise ValueError(
                "token dimension does not match metadata; expected "
                f"{self.metadata.feature_count()} got {tokens.shape[-1]}"
            )
        embeddings: list[torch.Tensor] = []
        for idx, feature in enumerate(self.metadata.features):
            table = self.tables[feature.name]
            embeddings.append(table(tokens[..., idx]))
        combined = torch.cat(embeddings, dim=-1)
        return self.dropout(combined)

    def zero_state(self, reference: torch.Tensor) -> torch.Tensor:
        shape = (*reference.shape[:-1], self.output_dim)
        return reference.new_zeros(shape)

