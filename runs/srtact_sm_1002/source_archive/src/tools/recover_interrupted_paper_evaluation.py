"""Recover only the evaluation phase of a fully trained interrupted paper run.

This utility deliberately refuses partially trained runs.  It first verifies every
periodic checkpoint, the source raw-step clock, CRCs, hashes recorded by the
training audit, and the final model clock.  Only then does it reproduce the same
deterministic final evaluation used by :mod:`tools.train_sb3`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sumo.paper_env import PaperSumoSceneEnv
from envs.sumo.paper_scenario_registry import PAPER_SCENARIOS
from envs.sumo.ppo_env import PaperPpoCarlaEnv, PaperPpoRgbEnv
from tools.paper_evaluation_contract import (
    build_evaluation_provenance,
    validate_model_environment_spaces,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _zip_data(path: Path) -> dict[str, object]:
    with ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Corrupt ZIP member {bad_member!r} in {path}")
        return json.loads(archive.read("data"))


def validate_completed_training(
    run_dir: Path, *, allow_existing_evaluation: bool = False
) -> dict[str, object]:
    """Return validated training metadata or reject an incomplete run."""

    run_dir = run_dir.resolve()
    arguments_path = run_dir / "arguments.json"
    audit_path = run_dir / "checkpoint_audit.json"
    final_model = run_dir / "final_model.zip"
    required = (arguments_path, audit_path, final_model)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Training is incomplete; missing {missing}")
    if not allow_existing_evaluation and (
        (run_dir / "final_evaluation.json").exists()
        or (run_dir / "paper_evaluation_detailed.json").exists()
    ):
        raise FileExistsError(
            f"Evaluation output already exists in {run_dir}; refusing to overwrite"
        )

    argument_payload = json.loads(arguments_path.read_text(encoding="utf-8"))
    arguments = argument_payload["requested_raw_steps"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    algo = str(arguments["algo"])
    scenario = str(arguments["scenario"])
    raw_steps = int(arguments["max_steps"])
    checkpoint_frequency = int(arguments["checkpoint_freq"])
    expected_clock_field = "num_timesteps" if algo == "ppo" else "_raw_steps_seen"
    expected_steps = list(range(checkpoint_frequency, raw_steps + 1, checkpoint_frequency))
    rows = audit.get("checkpoints", [])
    if not (
        str(audit.get("algorithm")) == algo
        and str(audit.get("scenario")) == scenario
        and int(audit.get("requested_raw_steps", -1)) == raw_steps
        and int(audit.get("checkpoint_frequency_raw_steps", -1))
        == checkpoint_frequency
        and str(audit.get("clock_field")) == expected_clock_field
        and int(audit.get("expected_checkpoint_count", -1)) == len(expected_steps)
        and len(rows) == len(expected_steps)
    ):
        raise ValueError(f"Checkpoint audit contract is incomplete in {run_dir}")

    for expected_step, row in zip(expected_steps, rows):
        checkpoint = run_dir / str(row.get("path", ""))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing audited checkpoint {checkpoint}")
        data = _zip_data(checkpoint)
        recorded_clock = int(data.get(expected_clock_field, -1))
        if not (
            int(row.get("raw_step", -1)) == expected_step
            and str(row.get("clock_field")) == expected_clock_field
            and int(row.get("expected_clock", -1)) == expected_step
            and int(row.get("recorded_clock", -1)) == expected_step
            and row.get("zip_crc_ok") is True
            and int(row.get("bytes", -1)) == checkpoint.stat().st_size
            and str(row.get("sha256", "")).upper() == _sha256(checkpoint)
            and recorded_clock == expected_step
        ):
            raise ValueError(
                f"Checkpoint {checkpoint} does not satisfy the exact source clock contract"
            )

    final_data = _zip_data(final_model)
    final_clock = int(final_data.get(expected_clock_field, -1))
    if final_clock != expected_steps[-1]:
        raise ValueError(
            f"Final model clock is {final_clock}, expected {expected_steps[-1]}"
        )
    if scenario not in PAPER_SCENARIOS:
        raise ValueError(f"Unknown paper scenario {scenario!r}")
    return {
        "arguments": arguments,
        "argument_payload": argument_payload,
        "audit_sha256": _sha256(audit_path),
        "final_model": final_model,
        "final_model_sha256": _sha256(final_model),
        "final_data": final_data,
    }


def recover_evaluation(run_dir: Path, *, device: str) -> dict[str, object]:
    # Keep the training-integrity validator importable without loading SB3's
    # optional tensorboard stack (and therefore an installed TensorFlow).
    from algos.sb3_torch import (
        SourcePPO,
        SceneRepresentationSAC,
        evaluate_model_detailed,
    )

    validated = validate_completed_training(run_dir)
    arguments = validated["arguments"]
    assert isinstance(arguments, dict)
    algo = str(arguments["algo"])
    scenario = str(arguments["scenario"])
    action_repeat = int(arguments["action_repeat"])
    discount = float(arguments["discount"])
    ego_control_profile = str(arguments.get("ego_control_profile", "direct"))
    traffic_protocol = str(arguments.get("traffic_protocol", "source_all"))
    traffic_partition = "all" if traffic_protocol == "source_all" else "evaluation"
    episode_limit_profile = str(arguments.get("episode_limit_profile", "source"))
    if algo == "ppo":
        env_class = PaperPpoCarlaEnv if scenario == "carla" else PaperPpoRgbEnv
        env = env_class(
            scenario=scenario,
            action_repeat=action_repeat,
            reward_discount=discount,
            training=False,
            ego_control_profile=ego_control_profile,
            traffic_partition=traffic_partition,
            episode_limit_profile=episode_limit_profile,
        )
    else:
        env = PaperSumoSceneEnv(
            scenario=scenario,
            neighbors=int(arguments["neighbors"]),
            history_steps=int(arguments["history_steps"]),
            path_length=int(arguments["path_length"]),
            action_repeat=action_repeat,
            reward_discount=discount,
            ego_control_profile=ego_control_profile,
            include_state_lstm=algo == "sac",
            state_lstm_only=algo == "sac",
            traffic_partition=traffic_partition,
            episode_limit_profile=episode_limit_profile,
        )
    model_class = {
        "scene_rep": SceneRepresentationSAC,
        "mst": SceneRepresentationSAC,
        "topo_scene": SceneRepresentationSAC,
        "topo_scene_balanced": SceneRepresentationSAC,
        "temporal_graph": SceneRepresentationSAC,
        "sac": SceneRepresentationSAC,
        "ppo": SourcePPO,
    }[algo]
    model_path = validated["final_model"]
    assert isinstance(model_path, Path)
    try:
        model = model_class.load(model_path, env=env, device=device)
        space_contract = validate_model_environment_spaces(model, env)
        trained_raw_steps = getattr(model, "_raw_steps_seen", None)
        if trained_raw_steps is None and isinstance(model, SourcePPO):
            trained_raw_steps = model.num_timesteps
        report = evaluate_model_detailed(
            model,
            env,
            episodes=int(arguments["eval_episodes"]),
            seed=int(arguments["seed"]) + 10_000,
            sumo_step_seconds=0.1,
            policy_action_hold=(3 if algo == "ppo" and scenario == "carla" else 1),
        )
        evaluation_provenance = build_evaluation_provenance(
            model=model,
            env=env,
            report=report,
            algorithm=algo,
            scenario=scenario,
            traffic_protocol=traffic_protocol,
            episode_limit_profile=episode_limit_profile,
            evaluation_seed_start=int(arguments["seed"]) + 10_000,
            environment_factory="tools.recover_interrupted_paper_evaluation",
            space_contract=space_contract,
        )
    finally:
        env.close()

    if int(trained_raw_steps) != int(arguments["max_steps"]):
        raise ValueError(
            f"Loaded final model has {trained_raw_steps} raw steps, expected "
            f"{arguments['max_steps']}"
        )
    selected_timesteps = int(model.num_timesteps)
    collected_training_raw_steps = (
        int(math.ceil(int(arguments["max_steps"]) / 512) * 512)
        if algo == "ppo"
        else int(arguments["max_steps"])
    )
    post_training_timesteps = (
        collected_training_raw_steps if algo == "ppo" else selected_timesteps
    )
    source_test_checkpoint_step = selected_timesteps if algo == "ppo" else None
    detailed = {
        "model": str(model_path.resolve()),
        "algorithm": algo,
        "scenario": scenario,
        "evaluation_seed_start": int(arguments["seed"]) + 10_000,
        "trained_raw_steps": int(trained_raw_steps),
        "collected_training_raw_steps": collected_training_raw_steps,
        "post_training_learner_timesteps": post_training_timesteps,
        "selected_model_learner_timesteps": selected_timesteps,
        "source_test_checkpoint_step": source_test_checkpoint_step,
        "evaluation_provenance": evaluation_provenance,
        "completion_time_population_std_ddof": 0,
        **report.to_dict(),
    }
    summary = detailed["summary"]
    (run_dir / "final_evaluation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (run_dir / "paper_evaluation_detailed.json").write_text(
        json.dumps(detailed, indent=2), encoding="utf-8"
    )
    recovery = {
        "kind": "evaluation_only_after_orchestrator_interruption",
        "training_was_not_resumed_or_modified": True,
        "run_directory": str(run_dir.resolve()),
        "checkpoint_audit_sha256": validated["audit_sha256"],
        "final_model_sha256": validated["final_model_sha256"],
        "evaluation_seed_start": detailed["evaluation_seed_start"],
        "episodes": summary["episodes"],
    }
    (run_dir / "interruption_recovery.json").write_text(
        json.dumps(recovery, indent=2), encoding="utf-8"
    )
    return {"recovery": recovery, "summary": summary}


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    result = recover_evaluation(args.run_dir.resolve(), device=args.device)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
