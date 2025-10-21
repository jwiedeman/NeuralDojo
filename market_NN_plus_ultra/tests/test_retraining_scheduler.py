from __future__ import annotations

from datetime import datetime, timedelta, timezone

import os
from pathlib import Path

import pytest

from market_nn_plus_ultra.automation import (
    DatasetStageConfig,
    RetrainingPlan,
    RetrainingScheduler,
    RetrainingSummary,
)
from market_nn_plus_ultra.automation.scheduler import DatasetSnapshot


def _make_summary(plan: RetrainingPlan, now: datetime) -> RetrainingSummary:
    return RetrainingSummary(plan=plan, stages=[], started_at=now, completed_at=now)


def test_scheduler_runs_when_dataset_updates(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.db"
    dataset_path.write_text("initial", encoding="utf-8")

    current_time = datetime(2025, 11, 23, 12, 0, tzinfo=timezone.utc)

    def clock() -> datetime:
        return current_time

    executed: list[RetrainingPlan] = []

    def plan_factory(snapshot: DatasetSnapshot) -> RetrainingPlan:
        plan = RetrainingPlan(
            dataset_path=snapshot.path,
            training_config=Path("train.yaml"),
            output_dir=tmp_path / "runs" / f"run_{len(executed)}",
            dataset_stage=DatasetStageConfig(strict_validation=False),
            run_pretraining=False,
            run_training=False,
        )
        executed.append(plan)
        return plan

    def executor(plan: RetrainingPlan) -> RetrainingSummary:
        return _make_summary(plan, clock())

    scheduler = RetrainingScheduler(
        dataset_path=dataset_path,
        plan_factory=plan_factory,
        executor=executor,
        poll_interval=1.0,
        clock=clock,
        sleep=lambda _: None,
    )

    summary1 = scheduler.check_and_run()
    assert summary1 is not None
    assert len(executed) == 1
    assert executed[0].dataset_path == dataset_path
    assert scheduler.history[-1][1] == summary1

    # No changes -> no additional run
    assert scheduler.check_and_run() is None

    # Modify dataset and advance time to guarantee mtime change
    current_time += timedelta(seconds=10)
    dataset_path.write_text("updated", encoding="utf-8")
    os.utime(dataset_path, None)

    current_time += timedelta(seconds=10)
    summary2 = scheduler.check_and_run()
    assert summary2 is not None
    assert len(executed) == 2
    assert executed[1].dataset_path == dataset_path


def test_scheduler_respects_cooldown(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data.db"
    dataset_path.write_text("seed", encoding="utf-8")

    current_time = datetime(2025, 11, 23, 13, 0, tzinfo=timezone.utc)

    def clock() -> datetime:
        return current_time

    run_count = 0

    def plan_factory(snapshot: DatasetSnapshot) -> RetrainingPlan:
        nonlocal run_count
        run_count += 1
        return RetrainingPlan(
            dataset_path=snapshot.path,
            training_config=Path("train.yaml"),
            output_dir=tmp_path / "sched_runs",
            dataset_stage=DatasetStageConfig(strict_validation=False),
            run_pretraining=False,
            run_training=False,
        )

    def executor(plan: RetrainingPlan) -> RetrainingSummary:
        return _make_summary(plan, clock())

    scheduler = RetrainingScheduler(
        dataset_path=dataset_path,
        plan_factory=plan_factory,
        executor=executor,
        poll_interval=1.0,
        cooldown_seconds=120.0,
        clock=clock,
        sleep=lambda _: None,
    )

    assert scheduler.check_and_run() is not None
    assert run_count == 1

    # Change occurs but within cooldown window -> run is deferred
    current_time += timedelta(seconds=30)
    dataset_path.write_text("change1", encoding="utf-8")
    os.utime(dataset_path, None)
    assert scheduler.check_and_run() is None
    assert run_count == 1

    # After cooldown expires the pending change should trigger exactly one run
    current_time += timedelta(seconds=200)
    assert scheduler.check_and_run() is not None
    assert run_count == 2

    # No additional run if nothing changed
    current_time += timedelta(seconds=10)
    assert scheduler.check_and_run() is None


def test_force_run_requires_existing_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "missing.db"

    current_time = datetime(2025, 11, 23, 14, 0, tzinfo=timezone.utc)

    def clock() -> datetime:
        return current_time

    def plan_factory(snapshot: DatasetSnapshot) -> RetrainingPlan:
        return RetrainingPlan(
            dataset_path=snapshot.path,
            training_config=Path("train.yaml"),
            output_dir=tmp_path,
            dataset_stage=DatasetStageConfig(strict_validation=False),
            run_pretraining=False,
            run_training=False,
        )

    def executor(plan: RetrainingPlan) -> RetrainingSummary:
        return _make_summary(plan, clock())

    scheduler = RetrainingScheduler(
        dataset_path=dataset_path,
        plan_factory=plan_factory,
        executor=executor,
        poll_interval=1.0,
        clock=clock,
        sleep=lambda _: None,
    )

    with pytest.raises(FileNotFoundError):
        scheduler.force_run()

    dataset_path.write_text("created", encoding="utf-8")
    summary = scheduler.force_run()
    assert summary.plan.dataset_path == dataset_path
    assert scheduler.history[-1][1] == summary
