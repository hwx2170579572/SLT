# Sorted scene: goal-only route topology and parameter-matched nonlinear 3-slot

Status: implementation smoke completed successfully on 2026-10-02; the formal 100k pair is running from `runs/sortg3_1002`. The smoke validates the integrated runner and diagnostics only and is not a performance result.

## Question

On the existing `intersection_sorted` task with `depart_scale=4.0`, compare two requested changes under one fixed seed and training/evaluation protocol:

1. `sac_mlp_d1_st_rt_topo_goalonly_v1`: preserve the existing STRT actor-intent and geometry-only relation path, and apply route reachability only to legal candidates in the goal-topology readout.
2. `sac_mlp_d1_st_rt_3slot_nonlinear_v1`: keep topology disabled and use a nonlinear three-slot readout with `ego 128→132→32`, `social 128→120→64`, and `route 128→132→32`, each with two ReLU activations. The three outputs concatenate to 128 dimensions. The requested active head size is 65,792 parameters, matching the existing joint readout count; matching parameter count does not imply matched FLOPs or inductive bias.

The route-aware method is a legal-candidate restriction, not a safety guarantee. The slot method tests a parameter-matched nonlinear head, not semantic slot disentanglement by itself.

## Shared protocol to lock before launch

- Scenario: `intersection_sorted`; explicit 30-template background pool with the existing three route counts (200, 150, 240) and `depart_scale=4.0`.
- Train seed: 0; fresh model, no resume; 100,000 raw SUMO control steps per method; action repeat 3; 5,000 raw-step learning warmup; checkpoint interval 10,000 raw steps.
- Final evaluation: deterministic 100-episode validation evaluation on logical seeds 10000–10099, final checkpoint only. Keep the same full effective 30-template traffic pool for train/evaluation as the prior sorted run protocol; do not silently switch to a holdout pool.
- Two independent Python worker processes share CUDA device 0. Evaluation and training diagnostics use the existing in-pipeline collectors. Evaluation must record shaped `env.step` reward and raw undiscounted reward separately, plus the existing component reconciliation fields.
- Terminal outcome semantics remain `exclusive_terminal_v2`. This is the current environment behavior; historical raw-return reports from earlier terminal/reward protocols are not directly comparable to new shaped returns.
- Collision-option evidence for these sorted D1 runs is inferred from the locked source construction chain `YieldObsIndependentV2EnvV1 -> IndependentV2FiveBySixEnvV1 -> HighDensityPaperSumoSceneEnvV1 -> PaperSumoSceneEnv`, not captured from the original child argv. `PaperSumoSceneEnv._sumo_command` supplies `--collision.action none` and `--collision.check-junctions true` for this non-random scene; the high-density layer replaces only `--route-files`. The static `envs/sumo/scenarios/intersection_sorted/scenario.sumocfg` value `collision.action=warn` is not evidence of the effective runtime option. This statement applies only to these sorted D1 methods; it does not revise older runs or DARRL settings.
- Preserve source files and hashes, sorted source/effective traffic hashes, arguments, worker PIDs/status, final checkpoint identity, and the final evaluation identity in a new short, non-existing run root. Do not reuse `runs/sort2_1001` or any prior root.

Expected per-method evidence includes `arguments.json`, `training_complete.json`, `final_model.zip`, `evaluation_results.json`, and train/eval diagnostic manifests, summaries, episode/decision/raw streams, optimization, and representation records. The run root also records the exact source archive, suite manifest/status, reward protocol validation, and pair comparison. These are planned artifacts, not evidence that a run has started.

## Evidence boundary and comparison

This is a two-method comparison with one training seed. It can distinguish the two specified configurations for this run, but cannot estimate training-seed variability. Keep the historical STRT, Topo, and 3-slot results as context only; those are not a concurrently rerun control in this launch. Report failures and missing outputs, and do not select a best checkpoint in place of the final checkpoint.

The route-aware variant must demonstrate that the goal-only reachability mask is enabled while actor-intent topology and relation-topology injection are disabled. The nonlinear-slot variant must demonstrate the exact three widths, two ReLUs per branch, 65,792 active head parameters, and disabled Topo/Graph-SLT/SBS auxiliary losses. Verify these in the saved method configuration and diagnostics before interpreting outcomes.

## Launch artifacts

The dedicated launcher is `../launch_sorted_goalonly_nonlinear3slot.py`. Preview mode does not create a run directory or environment. The smoke used a separate root and is implementation validation only; it must never be merged into the formal result table. The formal root was proposed as a fresh child under `runs/`; the actual root and launch snapshot are recorded below.

With the confirmed Conda interpreter, preview commands are:

```powershell
& 'D:\Programs\Anaconda\envs\pytorch\python.exe' 'D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\Scene-Rep-Transformer-main\pytorch_sb3_sumo\fast-developer\launch_sorted_goalonly_nonlinear3slot.py' --run-root 'D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\sortg3_1002'
& 'D:\Programs\Anaconda\envs\pytorch\python.exe' 'D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\Scene-Rep-Transformer-main\pytorch_sb3_sumo\fast-developer\launch_sorted_goalonly_nonlinear3slot.py' --smoke --run-root 'D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\sortg3sm_1002'
```

After the model/diagnostic smoke checks pass and the run is authorized, add `--start` to the second command for a 300-raw-step smoke, or use `--start --run-root 'D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\sortg3_1002'` for the formal pair. Smoke uses 60 raw warmup steps, 100-raw-step checkpoint frequency, and one evaluation episode per method; formal uses 5,000 raw warmup, 10,000-raw-step checkpoints, and 100 evaluation episodes. The two roots are independent; never promote smoke outputs into the formal root.

## Smoke validation snapshot (2026-10-02)

The smoke ran under `runs/sortg3sm_1002` and completed with launcher exit code 0 for both workers. Each method used `intersection_sorted`, `depart_scale=4.0`, seed 0, CUDA-configured execution, 300 raw steps, 241 updates, and one validation episode at seed 10000. For both methods, the `final_model.zip` SHA-256 matched `training_complete.json` and `evaluation_results.json`. Shaped and raw returns were recorded separately; the six reward components reconciled to the shaped return with maximum absolute error below `4e-15`.

Shadow diagnostics produced 3 unique training states and 2 unique evaluation states per method, with 12 probe rows per state. Both phases reported zero probe errors and zero active invalid rows; inactive branches were recorded as `branch_inactive`. Both smoke evaluation episodes ended in collision; this is implementation validation, not a performance result. Machine-readable evidence is in `sorted_goalonly_nonlinear3slot_smoke_validation_20261002.json`. The proposed formal root `runs/sortg3_1002` was absent at validation time. The Windows process command-line query was denied access, so process absence was not independently established by that query.

## Formal run snapshot (2026-10-02)

The formal pair is running from `runs/sortg3_1002`. Supervisor PID 38612 and coordinator PID 40248 launched two independent CUDA workers: PID 17464 for nonlinear 3-slot and PID 85576 for goal-only Topo. Both saved arguments specify fresh seed 0, 100000 raw steps, 5000 raw-step warmup, 10000 raw-step checkpoint interval, deterministic outer validation with 100 episodes, `intersection_sorted`/depart 4.0, behavior diagnostics and shadow probes enabled, train probe interval 5000 raw steps, and eval cap 3 states per episode. No resume option is present. At 09:26(+08), 3-slot had reached 5381 raw steps/381 updates and goal-only had reached 5082 raw/82 updates. Each train shadow stream had 1 unique state/12 variant rows; applicable rows were valid (8 and 7), inactive rows were marked separately (4 and 5), and active-invalid/exception counts were 0. Behavior diagnostics had written 5486/5332 raw records and 1834/1783 decision records, with zero diagnostic errors. At this snapshot neither method had a final checkpoint, training-complete marker, or evaluation result. The run source archive contains 807 source files and hashes for the 30 sorted templates. This is progress only; report performance after final evaluation and checkpoint identity checks. The application automation is ACTIVE and points to this run for completion follow-up. Machine-readable start snapshot: `sorted_goalonly_nonlinear3slot_formal_start_20261002.json`.
