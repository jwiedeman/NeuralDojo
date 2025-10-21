from pathlib import Path
import pytest

from market_nn_plus_ultra.training import (
    CurriculumCallback,
    CurriculumConfig,
    CurriculumParameters,
    CurriculumScheduler,
    CurriculumStage,
    DataConfig,
    summarise_curriculum_profile,
)


def make_data_config(**overrides):
    kwargs = dict(
        sqlite_path=Path("dummy.db"),
        window_size=128,
        horizon=4,
        stride=2,
        normalise=True,
    )
    kwargs.update(overrides)
    return DataConfig(**kwargs)


def test_curriculum_scheduler_progression():
    cfg = make_data_config()
    curriculum = CurriculumConfig(
        stages=[
            CurriculumStage(start_epoch=0, window_size=64, horizon=2, stride=1, normalise=False),
            CurriculumStage(start_epoch=5, window_size=192, horizon=4, stride=2),
        ],
        repeat_final=True,
    )
    scheduler = CurriculumScheduler(cfg, curriculum)

    params_epoch0 = scheduler.parameters_for_epoch(0)
    params_epoch3 = scheduler.parameters_for_epoch(3)
    params_epoch5 = scheduler.parameters_for_epoch(5)
    params_epoch10 = scheduler.parameters_for_epoch(10)

    assert params_epoch0 == CurriculumParameters(window_size=64, horizon=2, stride=1, normalise=False)
    assert params_epoch3 == params_epoch0  # still first stage
    assert params_epoch5 == CurriculumParameters(window_size=192, horizon=4, stride=2, normalise=True)
    assert params_epoch10 == params_epoch5  # repeats final stage


def test_curriculum_scheduler_errors_when_not_repeating_final_stage():
    cfg = make_data_config()
    curriculum = CurriculumConfig(
        stages=[CurriculumStage(start_epoch=0, window_size=32)],
        repeat_final=False,
    )
    scheduler = CurriculumScheduler(cfg, curriculum)

    with pytest.raises(ValueError):
        scheduler.parameters_for_epoch(10)


class DummyDataModule:
    def __init__(self):
        self.calls: list[int] = []

    def step_curriculum(self, epoch: int):
        self.calls.append(epoch)
        if epoch == 0:
            return None
        return CurriculumParameters(window_size=256, horizon=7, stride=3, normalise=True)


class DummyModule:
    def __init__(self) -> None:
        self.updated: list[int] = []
        self.logged: dict[str, float] = {}

    def update_horizon(self, horizon: int) -> None:
        self.updated.append(horizon)

    def log(self, key: str, value: float, **_: object) -> None:
        self.logged[key] = value


def test_curriculum_callback_triggers_updates():
    callback = CurriculumCallback()
    trainer = DummyTrainer()
    module = DummyModule()

    callback.on_train_epoch_start(trainer, module)

    assert trainer.datamodule.calls == [1]
    assert module.updated == [7]
    assert module.logged["curriculum/window_size"] == pytest.approx(256.0)
    assert module.logged["curriculum/horizon"] == pytest.approx(7.0)
    assert trainer.reset_train_calls == 1
    assert trainer.reset_val_calls == 1


class DummyTrainer:
    def __init__(self) -> None:
        self.datamodule = DummyDataModule()
        self.current_epoch = 1
        self.reset_train_calls = 0
        self.reset_val_calls = 0

    def reset_train_dataloader(self) -> None:
        self.reset_train_calls += 1

    def reset_val_dataloader(self) -> None:
        self.reset_val_calls += 1


def test_summarise_curriculum_profile_handles_missing_schedule():
    cfg = make_data_config()

    schedule = summarise_curriculum_profile(cfg, total_epochs=3)

    assert len(schedule) == 3
    expected = CurriculumParameters(
        window_size=cfg.window_size,
        horizon=cfg.horizon,
        stride=cfg.stride,
        normalise=cfg.normalise,
    )
    assert all(params == expected for params in schedule)


def test_summarise_curriculum_profile_resolves_stages():
    cfg = make_data_config(
        curriculum=CurriculumConfig(
            stages=[
                CurriculumStage(start_epoch=0, window_size=64, horizon=4, stride=2, normalise=False),
                CurriculumStage(start_epoch=2, window_size=128, horizon=6, stride=3, normalise=True),
            ],
            repeat_final=True,
        )
    )

    schedule = summarise_curriculum_profile(cfg, total_epochs=5)

    assert [params.window_size for params in schedule] == [64, 64, 128, 128, 128]
    assert [params.normalise for params in schedule] == [False, False, True, True, True]
