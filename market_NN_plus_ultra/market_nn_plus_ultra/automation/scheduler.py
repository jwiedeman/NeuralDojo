"""Scheduling utilities for continuous retraining workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
from threading import Event
import time
from typing import Callable, Optional

from .retraining import RetrainingPlan, RetrainingSummary, run_retraining_plan

LOGGER = logging.getLogger(__name__)


def _hash_file(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Return a BLAKE2 hash for ``path``."""

    hasher = hashlib.blake2b(digest_size=32)
    with path.open('rb') as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

def _default_clock() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


Clock = Callable[[], datetime]
SleepFn = Callable[[float], None]
PlanFactory = Callable[["DatasetSnapshot"], RetrainingPlan]
PlanExecutor = Callable[[RetrainingPlan], RetrainingSummary]


@dataclass(slots=True)
class DatasetSnapshot:
    """Lightweight metadata describing the state of a dataset artifact."""

    path: Path
    exists: bool
    size: int | None
    modified_at: datetime | None
    modified_ns: int | None
    fingerprint: str | None

    @classmethod
    def capture(cls, path: Path) -> "DatasetSnapshot":
        """Capture the current state of ``path`` as a :class:`DatasetSnapshot`."""

        path = Path(path)
        try:
            stat = path.stat()
        except FileNotFoundError:
            return cls(path=path, exists=False, size=None, modified_at=None, modified_ns=None, fingerprint=None)

        modified_ns = getattr(stat, "st_mtime_ns", None)
        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        fingerprint = _hash_file(path)
        return cls(
            path=path,
            exists=True,
            size=stat.st_size,
            modified_at=modified_at,
            modified_ns=modified_ns,
            fingerprint=fingerprint,
        )

    def signature(self) -> tuple[int | None, int | None, str | None]:
        """Return a tuple that uniquely identifies the snapshot."""

        return self.size, self.modified_ns, self.fingerprint


@dataclass(slots=True)
class RetrainingScheduler:
    """Poll a dataset for changes and trigger retraining plans when it updates."""

    dataset_path: Path
    plan_factory: PlanFactory
    executor: PlanExecutor = run_retraining_plan
    poll_interval: float = 300.0
    cooldown_seconds: float = 0.0
    clock: Clock = _default_clock
    sleep: SleepFn = time.sleep
    logger: logging.Logger = LOGGER
    history: list[tuple[datetime, RetrainingSummary]] = field(default_factory=list, init=False)
    last_snapshot: DatasetSnapshot | None = field(default=None, init=False)
    last_run_at: datetime | None = field(default=None, init=False)
    _pending_snapshot: DatasetSnapshot | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative")
        self.dataset_path = Path(self.dataset_path)

    # ------------------------------------------------------------------
    # Core scheduling logic
    # ------------------------------------------------------------------
    def _has_dataset_changed(self, snapshot: DatasetSnapshot) -> bool:
        if not snapshot.exists:
            return False
        if self.last_snapshot is None or not self.last_snapshot.exists:
            return True
        return snapshot.signature() != self.last_snapshot.signature()

    def _cooldown_active(self) -> bool:
        if self.cooldown_seconds <= 0 or self.last_run_at is None:
            return False
        elapsed = (self.clock() - self.last_run_at).total_seconds()
        if elapsed < self.cooldown_seconds:
            remaining = self.cooldown_seconds - elapsed
            self.logger.info(
                "Skipping retraining — cooldown active for %.2fs", remaining
            )
            return True
        return False

    def _resolve_pending_snapshot(
        self, current_snapshot: DatasetSnapshot
    ) -> tuple[DatasetSnapshot | None, bool]:
        if self._pending_snapshot is not None:
            if not current_snapshot.exists:
                # Dataset disappeared; drop pending work and propagate missing state.
                self.logger.warning(
                    "Dataset %s disappeared before retraining could run", self.dataset_path
                )
                self._pending_snapshot = None
                return None, False
            if current_snapshot.signature() != self._pending_snapshot.signature():
                self._pending_snapshot = current_snapshot
            return self._pending_snapshot, True

        if self._has_dataset_changed(current_snapshot):
            self._pending_snapshot = current_snapshot
            return current_snapshot, True

        return None, False

    def check_and_run(self) -> RetrainingSummary | None:
        """Check for dataset updates and run the plan when changes are detected."""

        current_snapshot = DatasetSnapshot.capture(self.dataset_path)
        snapshot_to_use, changed = self._resolve_pending_snapshot(current_snapshot)

        if not current_snapshot.exists:
            self.last_snapshot = current_snapshot
            return None

        if not changed:
            self.last_snapshot = current_snapshot
            return None

        if self._cooldown_active():
            return None

        assert snapshot_to_use is not None and snapshot_to_use.exists
        plan = self.plan_factory(snapshot_to_use)
        summary = self.executor(plan)
        completed_at = self.clock()
        self.last_run_at = completed_at
        self.last_snapshot = DatasetSnapshot.capture(self.dataset_path)
        self._pending_snapshot = None
        self.history.append((completed_at, summary))
        return summary

    # ------------------------------------------------------------------
    # Control helpers
    # ------------------------------------------------------------------
    def force_run(self) -> RetrainingSummary:
        """Trigger the retraining plan regardless of dataset state."""

        snapshot = DatasetSnapshot.capture(self.dataset_path)
        if not snapshot.exists:
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        plan = self.plan_factory(snapshot)
        summary = self.executor(plan)
        completed_at = self.clock()
        self.last_run_at = completed_at
        self.last_snapshot = snapshot
        self._pending_snapshot = None
        self.history.append((completed_at, summary))
        return summary

    def run_forever(
        self,
        *,
        stop_event: Event | None = None,
        max_iterations: Optional[int] = None,
    ) -> None:
        """Continuously poll until ``stop_event`` is set or ``max_iterations`` is reached."""

        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            if stop_event is not None and stop_event.is_set():
                break
            self.check_and_run()
            iterations += 1
            if stop_event is not None and stop_event.is_set():
                break
            self.sleep(self.poll_interval)


__all__ = [
    "DatasetSnapshot",
    "PlanFactory",
    "PlanExecutor",
    "RetrainingScheduler",
]
