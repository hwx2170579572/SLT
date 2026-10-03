"""Isolated entry point for the paired D1 C8/C9 contract repairs.

This module delegates training/evaluation orchestration to the established D1
runner, while restricting the registry to two fresh contract-fixed methods.
The entry point is a separate process boundary: existing D1 runs keep their
archived encoder and are never hot-patched.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
BASE_TRAINER_PATH = HERE / "train_intersection_yield_v2_d1.py"
METHODS = (
    "sac_mlp_d1_st_contractfix_v1",
    "sac_mlp_d1_st_rt_contractfix_v1",
)

for _path in (HERE, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_intersection_yield_v2_d1 as trainer  # noqa: E402
from algos.sb3_torch import incremental_topo_encoder as legacy_encoder  # noqa: E402
from algos.sb3_torch.contractfix_encoder import (  # noqa: E402
    ContractFixedIncrementalTopoEncoder,
)
from contractfix_trajectory_audit import TrajectoryHistoryAuditWrapper  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_contractfixed_constructor() -> None:
    """Inject the explicit SMARTS source contract at extractor construction.

    The established builder creates its ``features_extractor_kwargs`` locally.
    Its class import is module-based, so the adapter below adds the required
    source contract at that constructor boundary without copying the builder
    or changing the original training script.
    """
    cls = ContractFixedIncrementalTopoEncoder
    if not getattr(cls, "_trainer_smarts_contract_adapter", False):
        original_init = cls.__init__

        def init_with_smarts_contract(self, observation_space, **kwargs):
            supplied = kwargs.pop("source_observation_contract", None)
            if supplied not in (None, "smarts"):
                raise ValueError(
                    "Contract-fixed entry only accepts source_observation_contract='smarts'"
                )
            kwargs["source_observation_contract"] = "smarts"
            original_init(self, observation_space, **kwargs)

        cls.__init__ = init_with_smarts_contract
        cls._trainer_smarts_contract_adapter = True

    # _build_model_d1 imports this symbol inside the function. Replacing the
    # module export affects only this new worker process.
    legacy_encoder.IncrementalTopoEncoder = ContractFixedIncrementalTopoEncoder


def _add_method_registry() -> None:
    parent_config = trainer.D1_CONFIG["sac_mlp_d1_st"]
    route_config = trainer.D1_CONFIG["sac_mlp_d1_st_rt"]
    trainer.PARENT.update({method: "sac_mlp" for method in METHODS})
    trainer.D1_CONFIG[METHODS[0]] = {
        **parent_config,
        "method_protocol": "d1_c8_c9_contractfix_joint_v1",
        "contractfix_c8_last_valid_index": True,
        "contractfix_c9_cartesian_geometry_velocity": True,
    }
    trainer.D1_CONFIG[METHODS[1]] = {
        **route_config,
        "method_protocol": "d1_c8_c9_contractfix_joint_v1",
        "contractfix_c8_last_valid_index": True,
        "contractfix_c9_cartesian_geometry_velocity": True,
    }
    trainer.METHODS = tuple(trainer.PARENT)
    trainer.DISPATCH = {method: ("d1", None, method) for method in METHODS}
    trainer.ALL_METHODS = METHODS
    trainer.DEFAULT_METHODS = METHODS
    trainer.TRAIN_WORKERS = len(METHODS)


def _install_trajectory_audit() -> None:
    original_wrap = trainer._wrap_method_env

    def wrap_method_env(env, method: str, run_dir: Path, phase: str):
        env = original_wrap(env, method, run_dir, phase)
        if method not in METHODS:
            return env
        spaces = getattr(env.observation_space, "spaces", {})
        trajectory_space = spaces.get("trajectory")
        if trajectory_space is None:
            raise ValueError("Contract-fixed D1 requires observation key 'trajectory'")
        shape = tuple(int(value) for value in trajectory_space.shape)
        if len(shape) != 3 or shape[-1] != 5:
            raise ValueError(f"Expected raw [actors, history, 5] trajectory, got {shape}")
        # Both ST and ST-RT build explicit geometry-only vehicle edges.
        return TrajectoryHistoryAuditWrapper(
            env,
            run_dir=run_dir,
            phase=phase,
            method=method,
            geometry_active=True,
            trajectory_shape=shape,
            velocity_contract="smarts",
        )

    trainer._wrap_method_env = wrap_method_env


def _install_runtime_provenance() -> None:
    original_writer = trainer._write_runtime_provenance

    def write_contractfix_provenance(run_dir, method, **kwargs):
        record = original_writer(run_dir, method, **kwargs)
        if method not in METHODS:
            return record
        modules = {
            "contractfix_entrypoint": Path(__file__).resolve(),
            "legacy_d1_trainer": BASE_TRAINER_PATH.resolve(),
            "contractfix_encoder": Path(
                sys.modules["algos.sb3_torch.contractfix_encoder"].__file__
            ).resolve(),
            "trajectory_history_audit": Path(
                sys.modules["contractfix_trajectory_audit"].__file__
            ).resolve(),
            "topo_temporal_features": Path(
                sys.modules["algos.sb3_torch.topo_temporal_features"].__file__
            ).resolve(),
            "topo_temporal_features_v2": Path(
                sys.modules["algos.sb3_torch.topo_temporal_features_v2"].__file__
            ).resolve(),
            "features": Path(
                sys.modules["algos.sb3_torch.features"].__file__
            ).resolve(),
        }
        path = Path(run_dir) / "runtime_provenance.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        latest = payload["records"][-1]
        for label, source_path in modules.items():
            try:
                source_path.relative_to(PROJECT_ROOT)
            except ValueError as error:
                raise RuntimeError(
                    f"Runtime source escaped project root: {label}={source_path}"
                ) from error
            latest.setdefault("source_files", {})[label] = {
                "path": str(source_path),
                "sha256": _sha256(source_path),
            }
        latest["contractfix"] = {
            "method_protocol": "d1_c8_c9_contractfix_joint_v1",
            "c8": "max_true_valid_history_index_with_empty_history_handling",
            "c9": "SMARTS_pseudo_velocity_to_cartesian_only_for_geometry_edges",
            "source_observation_contract": "smarts",
            "source_contract_injection": "extractor_constructor_adapter_from_new_entrypoint",
            "geometry_active": True,
            "trajectory_audit": "every_real_preaction_decision; no replay or shadow forwards",
        }
        trainer.base._write_json_atomic(path, payload)
        return record

    trainer._write_runtime_provenance = write_contractfix_provenance


def install() -> None:
    if getattr(trainer, "_contractfix_entry_installed", False):
        return
    _install_contractfixed_constructor()
    _add_method_registry()
    _install_trajectory_audit()
    _install_runtime_provenance()
    # run_full uses the module's __file__ when it creates worker commands.
    # Point it at this immutable new entry, while retaining the original path
    # separately in runtime provenance and source archives.
    trainer.__file__ = str(Path(__file__).resolve())
    trainer._contractfix_entry_installed = True


def _require_guarded_run(argv=None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--help" in args or "-h" in args or "--make-scenario-only" in args or "--eval-only" in args:
        return
    # A fresh training invocation must come through the after-completion
    # launcher. Child workers inherit this exact root marker from run_full.
    if args and args[0] == "eval-worker":
        return
    forbidden_resume_flags = {
        "--resume", "--resume-from", "--continue-from", "--checkpoint-path",
        "--model-path",
    }
    if any(value in forbidden_resume_flags for value in args):
        raise RuntimeError("Contract-fixed training is fresh-only; resume/checkpoint input is forbidden")
    authorized_root = os.environ.get("D1_CONTRACTFIX_AUTHORIZED_RUN_ROOT")
    requested_root = None
    for index, value in enumerate(args[:-1]):
        if value == "--run-root":
            requested_root = args[index + 1]
            break
    if not authorized_root or not requested_root:
        raise RuntimeError(
            "Contract-fixed training can only start through the guarded launcher "
            "after the predecessor pair is complete."
        )
    if Path(authorized_root).expanduser().resolve() != Path(requested_root).expanduser().resolve():
        raise RuntimeError("Guarded run-root does not match this worker's --run-root")


def main(argv=None) -> int:
    _require_guarded_run(argv)
    install()
    trainer.__doc__ = __doc__
    return trainer.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
