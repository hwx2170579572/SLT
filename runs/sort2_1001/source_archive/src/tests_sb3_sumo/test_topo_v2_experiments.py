from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.run_topo_v2_experiments import (
    DEFAULT_CONTRACT,
    ExperimentContractError,
    accepted_run_reason,
    command_for,
    factorial_2x2,
    frozen_dependency_hashes,
    load_contract,
    make_jobs,
    run_directory,
    validate_contract,
)


class TopologyV2ExperimentContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = validate_contract(load_contract(DEFAULT_CONTRACT))
        cls.digest = "a" * 64

    def test_frozen_job_cardinalities(self):
        engineering = make_jobs(
            self.contract, "engineering", contract_digest=self.digest
        )
        factorial = make_jobs(
            self.contract, "causal_2x2", contract_digest=self.digest
        )
        lambdas = make_jobs(
            self.contract, "v5_lambda", contract_digest=self.digest
        )
        self.assertEqual(len(engineering), 8 * 6)
        self.assertEqual(len(factorial), 4 * 6 * 3)
        self.assertEqual(
            len(make_jobs(self.contract, "v2_merge", contract_digest=self.digest)),
            4 * 3,
        )
        self.assertEqual(len(lambdas), 3 * 3 * 3)
        self.assertEqual(len({job.name for job in engineering}), len(engineering))

    def test_soft_command_freezes_coefficient_and_validation_split(self):
        contract = copy.deepcopy(self.contract)
        jobs = make_jobs(
            contract,
            "v5_lambda",
            contract_digest=self.digest,
            methods="v5_soft_1e3",
            scenarios="carla",
            seeds="2",
        )
        command = command_for(
            contract, jobs[0], device="cuda", contract_digest=self.digest
        )
        self.assertEqual(command[command.index("--slot-balance-coef") + 1], "0.001")
        self.assertEqual(command[command.index("--evaluation-split") + 1], "validation")
        self.assertEqual(command[command.index("--max-steps") + 1], "100000")
        self.assertEqual(command[command.index("--episode-limit-profile") + 1], "source")

    def test_selected_candidate_cannot_be_resolved_before_freeze(self):
        contract = copy.deepcopy(self.contract)
        contract["methods"]["selected_candidate"]["resolver"] = str(
            Path(tempfile.gettempdir()) / "definitely_missing_topo_v2_candidate.json"
        )
        with self.assertRaises(ExperimentContractError):
            make_jobs(contract, "formal_test", contract_digest=self.digest)

    def test_engineering_receipt_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = copy.deepcopy(self.contract)
            contract["output_root"] = directory
            job = make_jobs(
                contract,
                "engineering",
                contract_digest=self.digest,
                methods="temporal_graph",
                scenarios="left_turn",
            )[0]
            root = run_directory(contract, job)
            root.mkdir(parents=True)
            arguments = {
                "experiment_contract_sha256": self.digest,
                "frozen_dependencies": frozen_dependency_hashes(contract),
                "requested_raw_steps": {
                    "algo": job.algorithm,
                    "scenario": job.scenario,
                    "seed": job.seed,
                    "max_steps": job.raw_steps,
                    "evaluation_split": job.evaluation_split,
                    "traffic_protocol": job.traffic_protocol,
                    "episode_limit_profile": job.episode_limit_profile,
                },
                "implementation_fidelity": {
                    "implementation_id": job.implementation_id
                },
                "effective_method_hyperparameters": {"slot_balance_coef": 0.0},
            }
            (root / "arguments.json").write_text(json.dumps(arguments))
            (root / "method_metadata.json").write_text(
                json.dumps({"implementation_id": job.implementation_id})
            )
            (root / "check_result.json").write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "action_finite": True,
                        "algorithm": job.algorithm,
                        "scenario": job.scenario,
                    }
                )
            )
            accepted, _ = accepted_run_reason(contract, job, self.digest)
            wrong_hash, _ = accepted_run_reason(contract, job, "b" * 64)
        self.assertTrue(accepted)
        self.assertFalse(wrong_hash)

    def test_factorial_effect_formulas(self):
        methods = {
            "temporal_graph": 0.10,
            "topology_v1": 0.30,
            "temporal_hard": 0.20,
            "full_hard": 0.70,
        }
        rows = []
        for scenario in (
            "left_turn",
            "cross",
            "roundabout_easy",
            "roundabout_medium",
            "roundabout",
            "carla",
        ):
            for seed in range(3):
                for method, success in methods.items():
                    rows.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "method": method,
                            "success_rate": success,
                            "collision_rate": 1.0 - success,
                            "mean_return": success,
                            "timeout_rate": 0.0,
                        }
                    )
        result = factorial_2x2(rows)
        success = next(
            row
            for row in result["effects"]
            if row["scenario"] == "cross" and row["metric"] == "success_rate"
        )
        self.assertTrue(result["complete"])
        self.assertAlmostEqual(success["mean_effects"]["topology_main_effect"], 0.35)
        self.assertAlmostEqual(success["mean_effects"]["hard_norm_main_effect"], 0.25)
        self.assertAlmostEqual(success["mean_effects"]["interaction"], 0.30)


if __name__ == "__main__":
    unittest.main()
