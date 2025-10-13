"""Structured logging helpers for Market NN Plus Ultra."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping


def _stringify(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return json.dumps([_stringify(v) for v in value])
    if isinstance(value, Mapping):
        return json.dumps({str(k): _stringify(v) for k, v in value.items()})
    return str(value)


@dataclass(slots=True)
class StructuredLogger:
    """Light-weight wrapper emitting JSON-formatted log messages."""

    _logger: logging.Logger
    _bound_fields: MutableMapping[str, Any] = field(default_factory=dict)

    def bind(self, **fields: Any) -> "StructuredLogger":
        combined = dict(self._bound_fields)
        combined.update(fields)
        return StructuredLogger(self._logger, combined)

    def debug(self, event: str, **fields: Any) -> None:
        self._logger.debug(self._format(event, fields))

    def info(self, event: str, **fields: Any) -> None:
        self._logger.info(self._format(event, fields))

    def warning(self, event: str, **fields: Any) -> None:
        self._logger.warning(self._format(event, fields))

    def error(self, event: str, **fields: Any) -> None:
        self._logger.error(self._format(event, fields))

    def _format(self, event: str, fields: Mapping[str, Any]) -> str:
        payload = {"event": event}
        payload.update({str(k): _stringify(v) for k, v in self._bound_fields.items()})
        payload.update({str(k): _stringify(v) for k, v in fields.items()})
        return json.dumps(payload, ensure_ascii=False)


def get_structured_logger(name: str, level: int = logging.INFO) -> StructuredLogger:
    """Return a :class:`StructuredLogger` configured for console JSON output."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return StructuredLogger(logger)


__all__ = [
    "StructuredLogger",
    "get_structured_logger",
]
