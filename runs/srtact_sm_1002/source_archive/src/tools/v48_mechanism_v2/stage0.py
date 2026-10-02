"""Historical diagnostics gate, distinct from the future causal-training gate."""
from __future__ import annotations

import json
from pathlib import Path

from .common import cpu_environment
RESOURCES = cpu_environment()

import numpy as np

from .common import ROOT, OUTPUT, SCENES, metadata, configure_torch, describe_file, load_json, write_json, probe_dataset_path
from .models import checkpoint_audit


def main():
    runtime = configure_torch()
    models, datasets = [], []
    for scene in SCENES:
        path = probe_dataset_path(scene)
        with np.load(path, allow_pickle=False) as data:
            schema = {k: dict(shape=list(data[k].shape), dtype=str(data[k].dtype)) for k in data.files}
            meta = json.loads(str(data["metadata"]))
            obs = {k: data[k][:4] for k in ("trajectory", "map")}
            datasets.append(dict(scene=scene, file=describe_file(path), schema=schema, metadata=meta,
                missing_control_fields=["action", "next_observation", "episode_outcome", "raw_time", "lane_action_mask"],
                grouping="episode_id", observed_episodes=len(np.unique(data["episode_id"]))))
        for name in ("mst_slt", "v4_8_lr_half", "late_decay", "batch64"):
            result = checkpoint_audit(scene, name, obs)
            models.append(result)
            write_json(OUTPUT / "stage0/models" / f"{scene}__{name}.json", result)
            print(f"PASS {scene}/{name}: CPU final actor; stop-gradient; optimizer ownership", flush=True)
    replay = load_json(OUTPUT / "stage0/replay_checks.json")
    assert replay["all_checks_passed"]
    import xml.etree.ElementTree as ET
    tests = ET.parse(OUTPUT / "stage0/pytest_smoke.xml").getroot()
    suites = list(tests.iter("testsuite"))
    assert sum(int(s.attrib.get("failures", 0)) + int(s.attrib.get("errors", 0)) for s in suites) == 0
    source_paths = ["algos/sb3_torch/sac.py", "algos/sb3_torch/sac_v2.py", "algos/sb3_torch/sac_v4.py",
                    "algos/sb3_torch/representation.py", "algos/sb3_torch/graph_representation.py",
                    "algos/sb3_torch/replay_buffer.py", "algos/sb3_torch/replay_buffer_v4_7.py",
                    "envs/sumo/paper_env.py", "envs/sumo/paper_env_v4.py", "tools/run_latent_probes.py"]
    findings = [
        dict(id="F01", finding="Historical MST uses four-step reward aggregation with gamma bootstrap; full uses actual gamma**h and n=16.", implication="minus_horizon is not a pure length ablation; all future causal models require a matched corrected return protocol."),
        dict(id="F02", finding="Full combines TTG-V2, Graph-SLT, hybrid action/entropy, and slot balance coefficient 0.01.", implication="Whole-model probes cannot isolate TTG or Graph-SLT; the B bridge must keep V2 and make slot regularization explicit."),
        dict(id="F03", finding="Cross auxiliary target encoder is critic_target; CARLA uses the online critic under no_grad. Generic SLT alone has a separate Cross target projector.", implication="Target update rule is a scenario/module confound; tau may change representation targets."),
        dict(id="F04", finding="Both rewards are terminal dominated; Cross source timeout truncates and CARLA source timeout terminates.", implication="Do not claim Cross sparse versus CARLA dense reward. Terminal semantics affect return/bootstrap interpretation."),
        dict(id="F05", finding="Legacy data omit actions, outcomes, next observations, physical masks and exact time. The minimum_ttc label is closest-approach time, not verified collision TTC.", implication="Reuse for local readability only; add timestamped diagnostic rollouts for future-control questions."),
        dict(id="F06", finding="Actor does not update shared encoder; TD and representation objectives both do. Auxiliary target path is stopped.", implication="Gradient compatibility diagnostics should compare TD and representation on shared encoder parameters."),
        dict(id="F07", finding="n-step code lacks intermediate entropy accumulation and off-policy importance ratios; terminal transition duplication is inherited.", implication="gamma**h alone does not establish a complete soft n-step/Retrace guarantee."),
        dict(id="F08", finding="Historical tests and stage1 seeds are already development data; all policies are trained with seed 0.", implication="Episode intervals describe conditional diagnostic uncertainty, not training-seed robustness.")]
    report = dict(**metadata(), runtime=runtime, resource_policy=RESOURCES, models=models, datasets=datasets,
                  findings=findings, code_fingerprints=[describe_file(ROOT / p) for p in source_paths],
                  correctness=dict(replay_checks_passed=len(replay["checks"]), pytest_passed=sum(int(s.attrib["tests"]) for s in suites)),
                  historical_diagnostics_gate="PASS_WITH_DECLARED_LIMITATIONS",
                  future_controlled_training_gate="NOT_READY: freeze U; implement and verify common A/B/C interfaces and corrected-return/regularizer contracts before policy training")
    write_json(OUTPUT / "stage0/audit.json", report)
    print(report["historical_diagnostics_gate"], flush=True)


if __name__ == "__main__":
    main()
