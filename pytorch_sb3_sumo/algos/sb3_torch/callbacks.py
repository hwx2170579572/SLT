"""Callbacks that keep SB3 accounting in original raw simulator steps."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from .evaluation import source_evaluation_augmentation


class SourceEvaluationCallback(EvalCallback):
    """Run periodic evaluation with the release's test-time augmentation flag.

    ``model.predict()`` switches PyTorch modules to evaluation mode, but the
    released Scene-Rep random rotation is controlled by an independent flag.
    The TensorFlow test configuration disables that flag explicitly, so SB3's
    periodic evaluation must do the same and restore it afterwards.
    """

    def __init__(
        self,
        *args: Any,
        evaluation_seed_start: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.evaluation_seed_start = (
            int(evaluation_seed_start)
            if evaluation_seed_start is not None
            else None
        )

    def _on_step(self) -> bool:
        evaluation_due = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if evaluation_due and self.evaluation_seed_start is not None:
            # Gymnasium consumes this seed on the first reset. Subsequent
            # episode seeds come from the same deterministically reset RNG
            # stream, so every checkpoint sees an identical simulation-seed
            # sequence. Historical checkpoints are additionally re-evaluated
            # with explicit per-episode seeds for final learning curves.
            self.eval_env.seed(self.evaluation_seed_start)
        with source_evaluation_augmentation(self.model):
            return super()._on_step()


class RawStepControlCallback(BaseCallback):
    """Enforce raw-step budget, warm-up updates, and checkpoint thresholds.

    SB3 counts one call to ``env.step`` while the released runner counts every
    0.1-second simulator step inside a three-step held action.  This callback
    bridges those clocks without changing the environment action contract.
    """

    def __init__(
        self,
        *,
        raw_step_budget: int,
        checkpoint_frequency: int = 0,
        checkpoint_path: str | Path | None = None,
        checkpoint_prefix: str = "model",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if raw_step_budget <= 0:
            raise ValueError("raw_step_budget must be positive")
        if checkpoint_frequency < 0:
            raise ValueError("checkpoint_frequency cannot be negative")
        self.raw_step_budget = int(raw_step_budget)
        self.checkpoint_frequency = int(checkpoint_frequency)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint_prefix = checkpoint_prefix
        self.total_raw_steps = 0
        self._next_checkpoint = self.checkpoint_frequency
        self._pending_checkpoints: list[int] = []

    def _on_training_start(self) -> None:
        self.training_env.env_method(
            "set_lifetime_raw_step_budget", self.raw_step_budget
        )

    def _on_rollout_start(self) -> None:
        self._save_pending_checkpoints()

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals["infos"]
        executed = sum(int(info.get("raw_steps_executed", 1)) for info in infos)
        if executed == 0 and self.total_raw_steps >= self.raw_step_budget:
            # SB3 checks its loop condition before env.step(). One zero-raw
            # sentinel call lets the preceding budget-ending transition be
            # stored and trained, then stops without advancing the simulator.
            self.model.num_timesteps -= len(infos)
            return False
        if executed <= 0:
            raise RuntimeError("Environment reported no executed raw simulator step")
        previous = self.total_raw_steps
        self.total_raw_steps = min(
            self.raw_step_budget, self.total_raw_steps + executed
        )

        crossed_checkpoints: list[int] = []
        while (
            self.checkpoint_frequency > 0
            and self._next_checkpoint <= self.total_raw_steps
        ):
            crossed_checkpoints.append(self._next_checkpoint)
            self._next_checkpoint += self.checkpoint_frequency

        # A source checkpoint can fall inside one three-step held action. The
        # released runner saves immediately after that raw step's update, not
        # after the rest of the held action. Let the source-clock SAC split its
        # intermediate updates and save at those exact points. A threshold at
        # the current boundary is deferred until SB3 stores the transition and
        # performs the final post-store update.
        mid_interval_checkpoints = [
            raw_step
            for raw_step in crossed_checkpoints
            if raw_step < self.total_raw_steps and self.checkpoint_path is not None
        ]
        if hasattr(self.model, "set_raw_step_interval"):
            self.model.set_raw_step_interval(
                previous,
                self.total_raw_steps,
                checkpoint_steps=mid_interval_checkpoints,
                checkpoint_callback=self._save_checkpoint,
            )
        elif mid_interval_checkpoints:
            raise RuntimeError(
                "Exact mid-action raw-step checkpoints require "
                "set_raw_step_interval() support"
            )
        self.logger.record("time/raw_simulation_steps", self.total_raw_steps)

        self._pending_checkpoints.extend(
            raw_step
            for raw_step in crossed_checkpoints
            if raw_step == self.total_raw_steps
        )

        return True

    def _on_training_end(self) -> None:
        self._save_pending_checkpoints()

    def _save_pending_checkpoints(self) -> None:
        if not self._pending_checkpoints or self.checkpoint_path is None:
            return
        for raw_step in self._pending_checkpoints:
            self._save_checkpoint(raw_step)
        self._pending_checkpoints.clear()

    def _save_checkpoint(self, raw_step: int) -> None:
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.model.save(
            self.checkpoint_path
            / f"{self.checkpoint_prefix}_raw_{raw_step}_steps"
        )


class BestTrainingSuccessCallback(BaseCallback):
    """Prospectively retain the policy with best rolling training success.

    The paper says that testing uses the policy with highest training success,
    but the released repository contains no executable saver for that rule.
    This transparent reconstruction uses the source logger's fixed denominator
    of 20 and starts only after episode 20, matching the location of the
    release's commented best-training checkpoint condition. Ties keep the
    earlier checkpoint because the unpublished tie rule cannot be recovered.
    """

    def __init__(self, output_path: str | Path, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.output_path = Path(output_path)
        self.successes: deque[int] = deque(maxlen=20)
        self.episodes_completed = 0
        self.best_success_rate = -1.0

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        infos: list[dict[str, Any]] = self.locals["infos"]
        for done, info in zip(dones, infos):
            if not bool(done):
                continue
            if "is_success" not in info:
                raise RuntimeError(
                    "Training environment ended an episode without is_success"
                )
            self.episodes_completed += 1
            self.successes.append(int(bool(info["is_success"])))
            success_rate = sum(self.successes) / 20.0
            self.logger.record("rollout/source_success_rate_last_20", success_rate)
            if (
                self.episodes_completed > 20
                and success_rate > self.best_success_rate
            ):
                self.best_success_rate = success_rate
                self._save(success_rate)
        return True

    def _save(self, success_rate: float) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(self.output_path)
        if hasattr(self.model, "_raw_steps_seen"):
            clock_field = "_raw_steps_seen"
            clock_value = int(self.model._raw_steps_seen)
        else:
            clock_field = "num_timesteps"
            clock_value = int(self.model.num_timesteps)
        metadata = {
            "selection_rule": "highest_training_success_rate_last_20",
            "fixed_denominator": 20,
            "minimum_completed_episodes_exclusive": 20,
            "tie_break": "earliest_strict_improvement",
            "episodes_completed": self.episodes_completed,
            "success_rate_last_20": success_rate,
            "clock_field": clock_field,
            "clock_value": clock_value,
            "model": str(self.output_path.with_suffix(".zip").resolve()),
            "paper_rule_reconstruction": True,
        }
        self.output_path.with_name("best_training_success.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
