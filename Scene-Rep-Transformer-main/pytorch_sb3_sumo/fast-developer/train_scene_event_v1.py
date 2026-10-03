"""Independent entry point. Never dispatches or edits legacy method entries."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from scene_event.protocol import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "smoke", "train"), default="check")
    parser.add_argument("--method", choices=("sac_scene_dualgraph_v1", "sac_scene_eventgraph_cv_v1"), required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--raw-steps", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--config-out", type=Path)
    parser.add_argument("--acceptance-receipt", type=Path)
    args = parser.parse_args()
    config = ExperimentConfig(method=args.method, seed=args.seed, device=args.device)
    if args.mode == "smoke":
        config = replace(config, raw_steps=1200, eval_episodes=2, learning_starts_raw=128)
    if args.raw_steps is not None:
        config = replace(config, raw_steps=args.raw_steps)
    if args.eval_episodes is not None:
        config = replace(config, eval_episodes=args.eval_episodes)
    config.validate()
    if args.config_out:
        from scene_event.provenance import save_json
        if args.config_out.exists():
            raise FileExistsError(args.config_out)
        save_json(args.config_out, config.to_dict())
    if args.mode == "check":
        print(json.dumps({"configuration_only": True, "config": config.to_dict(),
                          "sha256": config.digest(), "simulation_started": False}, indent=2))
        return
    if args.run_root is None:
        parser.error("--run-root is required for a run; existing roots cannot be reused")
    if args.mode == "train" and (config.raw_steps != 100000 or config.eval_episodes != 100):
        parser.error("Formal development runs require 100000 raw steps and 100 final evaluation episodes; use smoke for bounded checks")
    if args.mode == "train":
        if args.acceptance_receipt is None:
            parser.error("Formal runs require a fresh --acceptance-receipt from validate_scene_event_v1.py")
        from scene_event.acceptance import validate_receipt
        validate_receipt(args.acceptance_receipt, config.method)
    from scene_event.trainer import run
    result = run(config, args.run_root, kind="smoke" if args.mode == "smoke" else "development")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
