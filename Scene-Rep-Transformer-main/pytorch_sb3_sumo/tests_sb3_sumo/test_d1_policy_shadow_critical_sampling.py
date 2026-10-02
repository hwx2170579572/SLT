from __future__ import annotations

import importlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _runner(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "fast-developer"))
    return importlib.import_module("train_intersection_yield_v2_d1")


def _candidate(step):
    return {
        "decision_step": step,
        "pre_obs_raw_step": (step - 1) * 3,
        "observation": {"state": step},
    }


def test_critical_events_keep_unknown_route_and_missing_ttc_unknown(monkeypatch):
    runner = _runner(monkeypatch)
    events, status, context = runner._policy_shadow_critical_events(
        {
            "current_lane_can_reach_next_edge": None,
            "expected_target_lane_can_reach_next_edge": None,
            "lane_command_requested": -1,
        },
        {"min_cv_obb_ttc_s": None},
        previous_route_status=True,
    )
    assert events == []
    assert status is None
    assert context["current_lane_can_reach_next_edge"] is None
    assert context["action_window_min_cv_obb_ttc_s"] is None


def test_critical_events_label_route_request_transition_and_window_ttc(monkeypatch):
    runner = _runner(monkeypatch)
    events, status, context = runner._policy_shadow_critical_events(
        {
            "current_lane_can_reach_next_edge": True,
            "expected_target_lane_can_reach_next_edge": False,
            "lane_command_requested": -1,
            "expected_target_lane_id": "edge_0",
        },
        {"min_cv_obb_ttc_s": 3.0},
        previous_route_status=True,
    )
    assert events == ["known_unreachable_target_lane_requested", "action_window_ttc_le_3s"]
    assert status is True
    assert context["action_window_min_cv_obb_ttc_s"] == 3.0
    assert context["ttc_timing"] == "post_action_window_min_ttc"

    transition, status, _ = runner._policy_shadow_critical_events(
        {"current_lane_can_reach_next_edge": False},
        {"min_cv_obb_ttc_s": 3.0001},
        previous_route_status=True,
    )
    assert transition == ["route_eligible_to_ineligible_post_transition"]
    assert status is False


def test_critical_candidate_budget_preserves_initial_critical_and_preterminal(monkeypatch):
    runner = _runner(monkeypatch)
    candidates = {
        "first_valid_multi_car": _candidate(1),
        "first_valid_route_context": _candidate(2),
    }
    selected = runner._policy_shadow_select_eval_candidates(
        candidates,
        _candidate(8),
        _candidate(4),
        max_samples=3,
        critical_enabled=True,
    )
    assert [row["decision_step"] for row in selected] == [1, 4, 8]
    assert selected[0]["triggers"] == ["first_valid_multi_car"]
    assert selected[1]["triggers"] == ["first_critical_event"]
    assert selected[2]["triggers"] == ["last_preterminal_prediction"]


def test_cap4_reserves_distinct_route_and_ttc_critical_states(monkeypatch):
    runner = _runner(monkeypatch)
    selected = runner._policy_shadow_select_eval_candidates(
        {
            "first_valid_multi_car": _candidate(1),
            "first_valid_route_context": _candidate(2),
        },
        _candidate(9),
        _candidate(4),
        max_samples=4,
        critical_enabled=True,
        route_critical_candidate=_candidate(4),
        ttc_critical_candidate=_candidate(6),
    )
    assert [row["decision_step"] for row in selected] == [1, 4, 6, 9]
    assert [row["triggers"] for row in selected] == [
        ["first_valid_multi_car"],
        ["first_route_critical_event"],
        ["first_ttc_critical_event"],
        ["last_preterminal_prediction"],
    ]


def test_cap4_merges_route_and_ttc_when_the_same_action_triggers_both(monkeypatch):
    runner = _runner(monkeypatch)
    same_critical = _candidate(4)
    selected = runner._policy_shadow_select_eval_candidates(
        {"first_valid_multi_car": _candidate(1)},
        _candidate(8),
        _candidate(4),
        max_samples=4,
        critical_enabled=True,
        route_critical_candidate=same_critical,
        ttc_critical_candidate=same_critical,
    )
    assert [row["decision_step"] for row in selected] == [1, 4, 8]
    assert selected[1]["triggers"] == [
        "first_route_critical_event",
        "first_ttc_critical_event",
    ]


def test_same_decision_merges_trigger_labels_and_default_selector_stays_legacy(monkeypatch):
    runner = _runner(monkeypatch)
    same = _candidate(1)
    selected = runner._policy_shadow_select_eval_candidates(
        {
            "first_valid_multi_car": same,
            "first_valid_route_context": same,
        },
        _candidate(5),
        same,
        max_samples=3,
        critical_enabled=True,
    )
    assert len(selected) == 2
    assert selected[0]["triggers"] == [
        "first_valid_multi_car",
        "first_critical_event",
        "first_valid_route_context",
    ]
    assert selected[1]["triggers"] == ["last_preterminal_prediction"]

    legacy = runner._policy_shadow_select_eval_candidates(
        {
            "first_valid_multi_car": _candidate(1),
            "first_valid_route_context": _candidate(2),
        },
        _candidate(9),
        _candidate(4),
        max_samples=3,
        critical_enabled=False,
    )
    assert [row["decision_step"] for row in legacy] == [1, 2, 9]
    assert all("first_critical_event" not in row["triggers"] for row in legacy)


def test_critical_eval_flag_defaults_off_and_accepts_three_or_four_sample_budget(monkeypatch):
    import inspect

    runner = _runner(monkeypatch)
    assert inspect.signature(runner._evaluate_saved).parameters[
        "policy_shadow_critical_eval_sample"
    ].default is False
    assert inspect.signature(runner._evaluate_d1).parameters[
        "policy_shadow_critical_eval_sample"
    ].default is False
    with pytest.raises(ValueError, match="require policy shadow probes"):
        runner._evaluate_saved(
            "unused", Path("unused.zip"), Path("unused"), 1,
            policy_shadow_critical_eval_sample=True,
        )
    with pytest.raises(ValueError, match="cap of 3 or 4"):
        runner._evaluate_saved(
            "unused", Path("unused.zip"), Path("unused"), 1,
            behavior_diagnostics=True,
            policy_shadow_probes=True,
            policy_shadow_eval_max_per_episode=2,
            policy_shadow_critical_eval_sample=True,
        )
    # Cap=4 passes the sampler validation and reaches the mocked environment
    # factory, which deliberately stops before any environment is constructed.
    class ReachedEnvironmentFactory(RuntimeError):
        pass

    monkeypatch.setattr(
        runner.base,
        "make_env_factory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ReachedEnvironmentFactory("cap validation passed")
        ),
    )
    with pytest.raises(ReachedEnvironmentFactory, match="cap validation passed"):
        runner._evaluate_saved(
            "sac_mlp_d1_st_rt_topo_goalonly_v1",
            Path("unused.zip"),
            Path("unused"),
            1,
            behavior_diagnostics=True,
            policy_shadow_probes=True,
            policy_shadow_eval_max_per_episode=4,
            policy_shadow_critical_eval_sample=True,
        )


@pytest.mark.parametrize(
    ("shadow_enabled", "critical_enabled", "cap"),
    [(True, True, 3), (True, True, 4), (False, False, 3)],
    ids=["control-critical-shadow-cap3", "control-critical-shadow-cap4", "goaloff-shadow-disabled"],
)
def test_evaluate_saved_scopes_critical_holder_for_control_and_shadow_off_paths(
    monkeypatch, tmp_path, shadow_enabled, critical_enabled, cap
):
    """Exercise _evaluate_saved's finally path without constructing SUMO."""
    runner = _runner(monkeypatch)
    method = "sac_mlp_d1_st_rt_topo_goalonly_v1"
    parent = runner.PARENT[method]
    monkeypatch.setitem(runner.base.SEED_START, parent, 10_000)

    class DummyEnv:
        def __init__(self):
            self.unwrapped = self
            self.closed = False

        def close(self):
            self.closed = True

    class DummyRecorder:
        def __init__(self):
            self.decision = {}
            self.decision_count = 0
            self.raw_count = 0
            self.episode_index = 0
            self.calls = 0
            self.metadata = []

        def on_policy_step(self, action, reward, terminated, truncated, info=None, *args, **kwargs):
            self.calls += 1

        def write_run_metadata(self, metadata):
            self.metadata.append(metadata)

        def record_representation(self, **kwargs):
            raise AssertionError("empty fake encoder diagnostics should not be recorded")

    class DummyEncoder:
        def configure_diagnostics(self, enabled, **kwargs):
            self.configured = (enabled, kwargs)

        def diagnostic_values(self):
            return {}

    class DummyModel:
        def __init__(self):
            self.actor = SimpleNamespace(features_extractor=DummyEncoder())

        def predict(self, observation, *args, **kwargs):
            return np.array([0.25, 0.0], dtype=np.float32), None

    env = DummyEnv()
    recorder = DummyRecorder()
    model = DummyModel()
    monkeypatch.setattr(runner, "_env_adapter_d1", lambda _method: object())
    monkeypatch.setattr(runner, "_wrap_route_reachability_env", lambda value, _method: value)
    monkeypatch.setattr(
        runner.base,
        "make_env_factory",
        lambda *_args, **_kwargs: lambda *_a, **_kw: env,
    )
    monkeypatch.setattr(runner.base, "_environment_namespace", lambda: None)
    monkeypatch.setattr(
        runner.base,
        "_wrap_behavior_diagnostics_env",
        lambda value, *_args, **_kwargs: (value, recorder),
    )

    import algos.sb3_torch.evaluation as evaluation
    import algos.sb3_torch.sac as sac

    monkeypatch.setattr(
        sac.SceneRepresentationSAC,
        "load",
        classmethod(lambda _cls, *_args, **_kwargs: model),
    )
    monkeypatch.setattr(evaluation, "source_evaluation_augmentation", lambda _model: nullcontext())
    evaluator_recorder = {"value": recorder}

    def fake_evaluate(model_arg, _env, **_kwargs):
        if hasattr(model_arg, "predict"):
            model_arg.predict({"state": np.zeros((1,), dtype=np.float32)}, deterministic=True)
        active_recorder = evaluator_recorder["value"]
        # The policy input was captured at raw=0; the action-repeat window is
        # complete at raw=3 when the recorder's post-step hook runs.
        active_recorder.raw_count = 3
        active_recorder.decision_count = 1
        active_recorder.decision = {"min_cv_obb_ttc_s": 2.0}
        active_recorder.on_policy_step(
            np.array([0.25, 0.0]), 0.0, False, False,
            info={"current_lane_can_reach_next_edge": True,
                  "expected_target_lane_can_reach_next_edge": False,
                  "lane_command_requested": -1},
        )
        record = {
            "success": False,
            "collision": False,
            "off_route": False,
            "timeout": True,
            "episode_return": 0.0,
            "raw_episode_return": 0.0,
        }
        return SimpleNamespace(episode_records=[SimpleNamespace(to_dict=lambda: dict(record))])

    monkeypatch.setattr(evaluation, "evaluate_model_detailed", fake_evaluate)
    monkeypatch.setattr(runner, "_d1_evaluation_return_summary", lambda *_a, **_kw: {})
    sink_instances = []
    shadow_metadata = []

    class DummySink:
        def __init__(self, *_args, **_kwargs):
            self.rows_written = 0
            self.error_rows = 0
            self.closed = False
            sink_instances.append(self)

        def close(self, metadata=None):
            self.closed = True
            self.metadata = metadata
            return {"rows_written": self.rows_written}

    monkeypatch.setattr(runner, "_PolicyShadowProbeSink", DummySink)
    monkeypatch.setattr(runner, "_policy_shadow_eval_context", lambda _obs: {
        "valid_multi_car": True,
        "valid_route_context": True,
    })
    def capture_shadow_metadata(_sink, _model, _encoder, _observation, metadata, **_kwargs):
        shadow_metadata.append(metadata)
        _sink.rows_written += 1

    monkeypatch.setattr(runner, "_run_policy_shadow_sample", capture_shadow_metadata)

    original_callback = recorder.on_policy_step
    summary, records = runner._evaluate_saved(
        method,
        tmp_path / "frozen.zip",
        tmp_path,
        1,
        behavior_diagnostics=True,
        policy_shadow_probes=shadow_enabled,
        policy_shadow_eval_max_per_episode=cap,
        policy_shadow_critical_eval_sample=critical_enabled,
    )

    assert len(records) == summary["episodes"] == 1
    assert env.closed
    assert recorder.calls == 1
    assert recorder.on_policy_step == original_callback
    assert bool(sink_instances) is shadow_enabled
    if shadow_enabled:
        assert sink_instances[0].closed
    if critical_enabled:
        coverage = records[0]["policy_shadow_probe_coverage"]
        assert coverage["critical_event_counts"] == {
            "known_unreachable_target_lane_requested": 1,
            "action_window_ttc_le_3s": 1,
        }
        assert coverage["selector_status"]["critical_sampling_mode"] == (
            "split_route_ttc_cap4" if cap == 4 else "single_first_critical_cap3"
        )
        assert len(shadow_metadata) == 1
        timing = shadow_metadata[0]["critical_event_context"]
        assert timing["event_observation_timing"].startswith(
            "shadow candidate is the pre-action observation"
        )
        assert timing["source_action_pre_obs_raw_global_step"] == 0
        assert timing["source_action_pre_obs_raw_episode_step"] == 0
        assert timing["context_window_start_raw_global_step"] == 0
        assert timing["context_window_start_raw_episode_step"] == 0
        assert timing["context_window_end_raw_global_step"] == 3
        assert timing["context_window_end_raw_episode_step"] == 3
        assert shadow_metadata[0]["source_action_raw_window_end_global_step"] == 3
        assert shadow_metadata[0]["source_action_raw_window_end_episode_step"] == 3
