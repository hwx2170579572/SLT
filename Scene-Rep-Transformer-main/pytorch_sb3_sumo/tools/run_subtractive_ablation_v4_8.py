"""Train one subtractive-ablation arm of v4_8/lr_half, independent of the frozen phase-2 contracts.

Arms (each is a registered v4.8 algorithm id):
  minus_horizon    -> topo_v4_8_minus_horizon_correct_credit  (horizon-correct n_step 16 -> 4)
  minus_factorized -> topo_v4_8_minus_factorized_entropy       (factorized entropy -> joint entropy)

Protocol is pinned to lr_half (learning_rate=5e-5, tau=5e-3), seed 0, and reuses the SAME
scenario env args + deployment decoder as the v4_8 reference cell (plan.json source), so the
only difference between an arm and v4_8/lr_half is the subtracted mechanism.

Unlike tools/run_phase2_cell_v2.py, this runner does NOT read the frozen tuning_contract or
preflight, so it never mutates the phase-2 scope. Results land under results_subtractive_ablation/.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase1_checkpoint_diagnostics import read, write, sha, validate_report
from tools.phase2_model_factory_v1 import verify_optimizer_settings


ARMS = {
    "minus_horizon": "topo_v4_8_minus_horizon_correct_credit",
    "minus_factorized": "topo_v4_8_minus_factorized_entropy",
}
LEARNING_RATE = 5e-5   # lr_half protocol
TAU = 5e-3
RAW_BUDGET = 50_000
WARMUP = 5_000
BATCH_SIZE = 32
BUFFER_SIZE = 20_000
ACTION_REPEAT = 3
SEED = 0


def run(args: argparse.Namespace) -> None:
    import torch
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from algos.sb3_torch import RawStepControlCallback, evaluate_model_detailed
    from configs.sb3_configs_v4_8 import make_model_v4_8
    from tools.train_independent_v2_5m6s100e_v1 import _make_environment_factory
    from tools.paper_evaluation_contract import validate_model_environment_spaces

    torch.set_num_threads(1)
    algo = ARMS[args.arm]
    plan = read(ROOT / "results_phase1_checkpoint_diagnostics_v1/plan.json")
    # Reuse the v4_8 reference cell's env args + deployment decoder so the arm differs only structurally.
    source = next(
        s for s in plan["sources"]
        if s["method"] == "v4_8" and s["scenario"] == args.scenario
    )
    # Part B 已做：v4_8/lr_half 部署以「无选择器 + actor_deterministic」为主。
    # 消融臂用同一 actor 确定性解码，消除 checkpoint/decoder 选择器与融合门控的混杂。
    decoder = "actor_deterministic"
    requested = read(ROOT / source["run"] / "arguments.json")["requested_raw_steps"]
    env_args = argparse.Namespace(
        **dict(requested, seed=SEED, gui=False, evaluation_split="validation")
    )
    protocol = read(ROOT / plan["protocol"])
    namespace = hashlib.sha256(f"{algo}__{args.scenario}".encode()).hexdigest()[:12]

    raw_budget, warmup, batch, buffer = (
        (72, 48, 2, 128) if args.smoke else (RAW_BUDGET, WARMUP, BATCH_SIZE, BUFFER_SIZE)
    )
    stage = "smoke" if args.smoke else "screen"
    output = ROOT / "results_subtractive_ablation" / stage / f"{args.arm}__{args.scenario}__lr_half__seed{SEED}"
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite an interrupted or completed run.
    started = time.time()
    write(output / "status.json", dict(
        status="starting", smoke=args.smoke, algorithm=algo, scenario=args.scenario, device=args.device,
    ))

    factory = _make_environment_factory(
        adapter="v4_8",
        density=protocol["scenarios"][args.scenario],
        overlay_root=ROOT / "results_subtractive_ablation" / "ov" / namespace,
    )

    checks = []

    class OptimizerAudit(BaseCallback):
        def _on_step(self):
            verify_optimizer_settings(self.model, learning_rate=LEARNING_RATE, tau=TAU)
            if self.n_calls % 100 == 0:
                write(output / "progress.json", dict(
                    raw_steps=self.model._raw_steps_seen,
                    learner_updates=self.model._n_updates,
                    updated_at=time.time(),
                ))
            return True

        def _on_training_end(self):
            checks.append(verify_optimizer_settings(self.model, learning_rate=LEARNING_RATE, tau=TAU))

    env = None
    evaluation_env = None
    try:
        env = Monitor(
            factory(env_args),
            filename=str(output / "train_monitor.csv"),
            info_keywords=("raw_simulation_steps", "is_success", "collision", "off_route", "max_time"),
        )
        model = make_model_v4_8(
            algo,
            env,
            scenario=args.scenario,
            learning_rate=LEARNING_RATE,
            batch_size=batch,
            learning_starts=warmup,
            buffer_size=buffer,
            action_repeat=ACTION_REPEAT,
            seed=SEED,
            device=args.device,
            verbose=0,
            tensorboard_log=str(ROOT / "results_subtractive_ablation" / "tb" / namespace),
        )
        model.tau = TAU
        verify_optimizer_settings(model, learning_rate=LEARNING_RATE, tau=TAU)
        write(output / "arguments.json", dict(
            arm=args.arm,
            algorithm=algo,
            scenario=args.scenario,
            candidate="lr_half",
            learning_rate=LEARNING_RATE,
            tau=TAU,
            smoke=args.smoke,
            raw_budget=raw_budget,
            warmup=warmup,
            batch_size=batch,
            buffer_size=buffer,
            seed=SEED,
            decoder=decoder,
            device=args.device,
            model_metadata=dict(getattr(model, "v4_method_metadata", {}) or {}),
        ))
        model.learn(total_timesteps=raw_budget, callback=CallbackList([
            RawStepControlCallback(
                raw_step_budget=raw_budget,
                checkpoint_frequency=36 if args.smoke else 10_000,
                checkpoint_path=output / "checkpoints",
                checkpoint_prefix="ckpt",
            ),
            OptimizerAudit(),
        ]))
        assert model._raw_steps_seen == raw_budget
        assert model._n_updates == raw_budget - warmup + 1
        verify_optimizer_settings(model, learning_rate=LEARNING_RATE, tau=TAU)
        checkpoint = output / "final_model.zip"
        model.save(checkpoint)
        write(output / "training_diagnostics.json", model.training_diagnostics())
        write(output / "training_complete.json", dict(
            checkpoint_sha256=sha(checkpoint),
            raw_steps=model._raw_steps_seen,
            learner_updates=model._n_updates,
            optimizer_settings=verify_optimizer_settings(model, learning_rate=LEARNING_RATE, tau=TAU),
        ))

        evaluation_env = factory(env_args, evaluation=True)
        # 无选择器 + actor_deterministic：默认 policy_class 加载，再把 _predict 补丁为
        # 恒用 actor 确定性解码（等价 phase3 apply_decoder(..., 'actor_deterministic')）。
        restored = type(model).load(checkpoint, env=evaluation_env, device=args.device)
        import types as _types

        def _actor_deterministic_predict(self, observation, deterministic=False):
            return self.actor(observation, deterministic=True)

        restored.policy._predict = _types.MethodType(_actor_deterministic_predict, restored.policy)
        after = verify_optimizer_settings(restored, learning_rate=LEARNING_RATE, tau=TAU)
        assert restored._n_updates == model._n_updates
        assert restored._raw_steps_seen == model._raw_steps_seen
        for key, tensor in model.policy.state_dict().items():
            assert torch.equal(tensor.cpu(), restored.policy.state_dict()[key].cpu()), key
        validate_model_environment_spaces(restored, evaluation_env)

        from tools.phase1_checkpoint_diagnostics import summarize
        episodes = 1 if args.smoke else 100
        records = []
        reference = read(ROOT / source["run"] / "paper_evaluation_detailed.json")["episode_records"]
        for offset in range(0, episodes, 20):
            count = min(20, episodes - offset)
            d = evaluate_model_detailed(
                restored, evaluation_env, episodes=count, seed=10000 + offset,
                deterministic=True, sumo_step_seconds=0.1, policy_action_hold=1,
            ).to_dict()
            validate_report(d, count, 10000 + offset)
            assert [r["traffic_variant"] for r in d["episode_records"]] == [
                r["traffic_variant"] for r in reference[offset:offset + count]
            ]
            for r in d["episode_records"]:
                r["episode"] = r["seed"] - 10000
            records.extend(d["episode_records"])
            write(output / f"evaluation_block_{offset:03d}.json", dict(
                **d, checkpoint_sha256=sha(checkpoint), decoder=decoder, smoke=args.smoke,
            ))
        write(output / "evaluation.json", dict(
            **summarize(records), checkpoint_sha256=sha(checkpoint), decoder=decoder, smoke=args.smoke,
        ))
        write(output / "status.json", dict(
            status="completed", smoke=args.smoke, scientific_result=not args.smoke,
            raw_steps=model._raw_steps_seen, learner_updates=model._n_updates,
            optimizer_checks=checks, restored_optimizer_settings=after,
            policy_tensor_roundtrip_equal=True, evaluation_episodes=episodes,
            wall_seconds=time.time() - started,
        ))
        print(str(output), flush=True)
    except BaseException as exc:
        write(output / "status.json", dict(
            status="failed", smoke=args.smoke, error=repr(exc), wall_seconds=time.time() - started,
        ))
        raise
    finally:
        if evaluation_env is not None:
            evaluation_env.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
