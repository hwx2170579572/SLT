# Sorted scene: goal-only route topology and parameter-matched nonlinear 3-slot

Status: implementation smoke completed successfully on 2026-10-02; the formal 100k run has not started. The smoke validates the integrated runner and diagnostics only and is not a performance result.

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
- Preserve source files and hashes, sorted source/effective traffic hashes, arguments, worker PIDs/status, final checkpoint identity, and the final evaluation identity in a new short, non-existing run root. Do not reuse `runs/sort2_1001` or any prior root.

Expected per-method evidence includes `arguments.json`, `training_complete.json`, `final_model.zip`, `evaluation_results.json`, and train/eval diagnostic manifests, summaries, episode/decision/raw streams, optimization, and representation records. The run root also records the exact source archive, suite manifest/status, reward protocol validation, and pair comparison. These are planned artifacts, not evidence that a run has started.

## Evidence boundary and comparison

This is a two-method comparison with one training seed. It can distinguish the two specified configurations for this run, but cannot estimate training-seed variability. Keep the historical STRT, Topo, and 3-slot results as context only; those are not a concurrently rerun control in this launch. Report failures and missing outputs, and do not select a best checkpoint in place of the final checkpoint.

The route-aware variant must demonstrate that the goal-only reachability mask is enabled while actor-intent topology and relation-topology injection are disabled. The nonlinear-slot variant must demonstrate the exact three widths, two ReLUs per branch, 65,792 active head parameters, and disabled Topo/Graph-SLT/SBS auxiliary losses. Verify these in the saved method configuration and diagnostics before interpreting outcomes.

## Launch artifacts

The dedicated launcher is `../launch_sorted_goalonly_nonlinear3slot.py`. Preview mode must not create a run directory or environment. A smoke run, when authorized, must use a separate root and be marked as smoke; it is implementation validation only and must never be merged into the formal result table. The formal root is expected to be a fresh child under `runs/` (proposed `runs/sortg3_1002`, subject to an existence/path-length check immediately before launch).

With the confirmed Conda interpreter, preview commands are:

```powershell
& 'D:\Programs\Anaconda\envs\pytorch\python.exe' 'D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\Scene-Rep-Transformer-main\pytorch_sb3_sumo\fast-developer\launch_sorted_goalonly_nonlinear3slot.py' --run-root 'D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\sortg3_1002'
& 'D:\Programs\Anaconda\envs\pytorch\python.exe' 'D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\Scene-Rep-Transformer-main\pytorch_sb3_sumo\fast-developer\launch_sorted_goalonly_nonlinear3slot.py' --smoke --run-root 'D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\sortg3sm_1002'
```

After the model/diagnostic smoke checks pass and the run is authorized, add `--start` to the second command for a 300-raw-step smoke, or use `--start --run-root 'D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\runs\sortg3_1002'` for the formal pair. Smoke uses 60 raw warmup steps, 100-raw-step checkpoint frequency, and one evaluation episode per method; formal uses 5,000 raw warmup, 10,000-raw-step checkpoints, and 100 evaluation episodes. The two roots are independent; never promote smoke outputs into the formal root.

## Smoke validation snapshot (2026-10-02)

The smoke ran under `runs/sortg3sm_1002` and completed with launcher exit code 0 for both workers. Each method used `intersection_sorted`, `depart_scale=4.0`, seed 0, CUDA-configured execution, 300 raw steps, 241 updates, and one validation episode at seed 10000. For both methods, the `final_model.zip` SHA-256 matched `training_complete.json` and `evaluation_results.json`. Shaped and raw returns were recorded separately; the six reward components reconciled to the shaped return with maximum absolute error below `4e-15`.

Shadow diagnostics produced 3 unique training states and 2 unique evaluation states per method, with 12 probe rows per state. Both phases reported zero probe errors and zero active invalid rows; inactive branches were recorded as `branch_inactive`. Both smoke evaluation episodes ended in collision; this is implementation validation, not a performance result. Machine-readable evidence is in `sorted_goalonly_nonlinear3slot_smoke_validation_20261002.json`. The proposed formal root `runs/sortg3_1002` was absent at validation time. The Windows process command-line query was denied access, so process absence was not independently established by that query.
