# Topology-Temporal v2 completion audit

This is a live requirement-to-evidence ledger. It is deliberately excluded from
the immutable source/asset manifest so that status can be updated while frozen
runs are executing. `TBD` means that the required real experiment has not yet
finished; it never denotes an imputed result.

## Evidence identity

| Evidence | SHA-256 / result | Status |
| --- | --- | --- |
| Experiment contract | `fbabbade013ab2c9ae2a3c9a5b2a42b9c5d1751b244a40316790a739fff95e82` | frozen |
| Fast iteration contract | `42b6b32c343125ca4e97cbdb26019fa0bbf63b785edd13612b996a8710090455` | validated; separate from original |
| Source and asset manifest file | `5081576f7ee92d50478131c6be78878f47f42058029d2b5543c72702668bf850` | frozen |
| Six-scenario topology audit | `94b571e49cc0363b55c959437aa0097d1c457f0c49c4343e8d3230cf5c629ad4` | passed |
| Runtime environment receipt | `c947ab6848f9ea5db09d1afce374d8cb3d9a70f906418ee87d5b3750868321c3` | frozen |
| Full `tests_sb3_sumo` regression | pre-experiment `201 passed`; final `206 passed, 5 warnings, 99.97 seconds` | passed |
| Engineering method/scenario matrix | `48/48 accepted` | passed |

## Requirement coverage

| Original requirement | Authoritative evidence | Current finding |
| --- | --- | --- |
| Keep v1 intact and implement v2 as distinct files | `*_v2.py`, `experiments/topo_scene_v2/`, freeze manifest | proven |
| Use `llm_pipeline` and record dependencies/hardware | `artifacts/topo_v2/runtime_environment.json` and every run's `arguments.json` | proven |
| Add symmetric MERGE without silent truncation | `topology_graph_v2.py`, six-scenario audit, topology tests | proven structurally; screen-positive mechanism, but cumulative candidate failed promotion efficacy |
| Route/direction/distance query, reverse mask, compatible top-k and fallback | `topo_temporal_features_v2.py`, unit tests, run diagnostics | proven structurally; promotion localisation gate passed |
| Near-zero topology and goal residuals; zero-gate equivalence | v2 feature extractor and regression tests | proven structurally; residuals remained bounded in promotion |
| Separate route and topology pooling | v2 feature extractor and shape/serialization tests | proven structurally; did not prevent CARLA policy collapse |
| Replace hard per-sample normalisation with batch-statistic SoftBalancedSlots | `sac_v2.py`, v2 extractor, coefficient sweep contract | proven structurally; lambda 0.01 selected, then rejected at promotion |
| Do not change reward, action space, SLT target, actor/critic or traffic contents | both contract validators, per-run accepted receipt and frozen asset manifest | proven; fast screening changes only raw-step/evaluation budgets |
| Keep 100k for confirmatory testing | fast contract formal stage and hard promotion lock | proven structurally; formal run not started |
| First perform topology × hard-normalisation 2x2 | original stop snapshot plus fast `factorial_anchor` summary/attribution/decision | original matrix stopped at 4/72 accepted; fast screen is 8/8 accepted and deeply attributed |
| Cumulative V2→V3→V4→V5 single-change experiments | fast stage definitions, dependency checks and per-stage decisions | V2--V5 screens 15/15 complete; candidate frozen |
| Record Success, Collision, Return, Off-route, timeout and completion time | detailed evaluation plus phase summary schema | F1--F6 and promotion 12/12 recorded; formal intentionally absent after gate failure |
| Record route-compatible mass, effective lanes, MERGE attention, gates and slot RMS/latent std | training/action diagnostic schemas and accepted-run validation | F1--F6 and promotion diagnostics recorded; promotion mechanism gate passed |
| Record lateral action proportions, ±1/3 margin, applied lane change and speed | aligned single-rollout JSONL/action aggregate | F1--F6, promotion, and bounded v3 CARLA diagnostic recorded |
| Record parameters, training time, inference time and peak GPU memory | `performance_profile.json` and summary schema | F1--F6 and promotion 12/12 recorded; candidate cost exceeded comparator while efficacy gate failed |
| Rapid development after user scope change | fast contract: 20k one-seed screening and 50k two-seed paired promotion | screens 23/23 and promotion 12/12 completed; one zero-training v3 stop diagnostic rejected at CARLA stage A |
| Select lambda only on validation data and allow lambda=0/V4 | frozen selection rule and `select-candidate` command | `v5_soft_1e2` / lambda 0.01 frozen from validation only |
| Deep attribution from real effects and predeclared stop rules | F1--F6, promotion and v3 summaries, gates, traces and deep reports | localisation/balance passed; CARLA failure attributed to downstream state-conditioned action timing; scalar calibration rejected |
| Do not access formal test before development gate passes | split implementation, runner hard lock, tests, and negative launch receipt | runtime-proven: missing gate is rejected before stage/run directory creation; receipt SHA `84eea31b43395cae76c0653c22fad84beb28ade49d6b8b9a2938a36c87ccb3df`; formal test not started |
| Formal comparison uses 10 seeds × 50 episodes, paired hierarchical bootstrap and Holm correction | frozen contract, statistics implementation/tests, declaration preflight, and formal reporting preregistration | 120 unique declarations validated without test access; operational Holm family fixed at 12 tests (6 scenarios × 2 primary metrics); launch remains locked |
| Save all settings, raw episode/action records, results and receipts | required-artifact contract and accepted-run validator | all executed development/promotion cells complete; unexecuted formal cells explicitly absent, never imputed |
| Produce complete final tables and bounded claims without invented cells | promotion summary/CSV, deep reports, gate and detached receipts | completed for the executed evidence; bounded negative conclusion, no formal table because the development gate failed |

## Conditional follow-ups

The source proposal makes the following experiments conditional rather than part
of the initial v2 bundle:

1. A 200k/500k budget curve is run only if evidence suggests that extra budget,
   rather than the structural repair, explains recovery. Any such conclusion must
   be labelled training insufficiency at 100k.
2. A categorical lane-intent × continuous-speed head is considered only if action
   sign saturation remains after the representation repair. It must be a separate
   experiment and cannot be mixed into the frozen representation comparison.

The extra-budget trigger did not fire: the dominant failure is action timing,
not evidence of slow representation convergence. The action-path trigger did
fire. A zero-training scalar calibration was therefore tested under a separate,
preregistered two-stage stop rule and failed its first CARLA gate. A categorical
or feasibility-aware head would be a new training-time method family and was not
started in this bounded rapid-iteration cycle.

## Operational recovery rule

- An accepted run is skipped on relaunch only after every required receipt and
  hash has been revalidated.
- An existing incomplete or mismatched run directory is never overwritten by the
  phase runner.
- The 10k interval model archives are scientific audit/budget-curve checkpoints,
  not exact process-resume checkpoints: they do not bind a complete replay
  buffer, live SUMO state, and all RNG streams.
- After an external interruption, preserve the incomplete directory as failed-run
  evidence and rerun that job from raw step zero. Never combine its partial trace
  with the replacement run and never mark it accepted.

## Current gate state

- Original development stage: stopped by user scope change; 4 complete, 4 incomplete, 64 unstarted; no files deleted.
- Completed fast stage: `factorial_anchor`, 8/8 accepted; summary SHA `8755f799a5217eaca1672f62681fc2bb0af8dfd7acc4cadec3a210c2fe427ddc`.
- Completed fast stage: `merge_probe`, 3/3 accepted; summary SHA `4967f469405a1aa1386f9cbe40593a02d4391335d2b22a460b3808be47a258fe`.
- Completed fast stage: `query_probe`, 3/3 accepted; summary SHA `36ee44d9effe12375d4b6b617dfac46409237288519b39b94e7c46eac7c805e4`.
- Completed fast stage: `gated_probe`, 3/3 accepted; summary SHA `2abb39b5c3642a0857972515587b1b37c1ef88738a3214bb135270981bccc55f`.
- Completed fast stage: `soft_center`, 2/2 accepted; summary SHA `15dcf4df3880a1c0cd91085cc5dd86f47c376f6e0bcf54fb505111a2787047f6`.
- Completed fast stage: `soft_bracket`, 4/4 accepted; summary SHA `fff08f0a358babd2df7c034022dc7cd44622fe51610c7b045e842c93495d092a`.
- Frozen candidate: `v5_soft_1e2`, lambda `0.01`; selection SHA `424b60ea7397e602da7b768ccffd3d981fe7c844268eabf80f3c9302416b7b04`.
- Completed fast stage: paired `promotion`, 12/12 accepted with zero scheduler failures and zero stderr bytes; summary SHA `db8ce421da622e09919e541c240f8a6468231cf564d2894ec44307f72e2f2c8c`.
- Promotion gate: **failed**. Macro Success delta `-0.161111`, worst paired Success delta `-1.0`, and no scenario reached the frozen positive-effect threshold. Gate SHA: `4527894a32365d372bd5706d0b90a3412f9dae2349397495be60a6d558c6da8c`.
- Promotion mechanism gate: passed; minimum route-compatible mass `0.999463`, maximum fallback `0.000537`, maximum effective lanes `3.541154`. This does not compensate for efficacy failures.
- Promotion deep attribution SHA: `95106a6dba078206bb684a232e8d5299370fdda17f0e0b6e9c640e0249f925c5`; detached completion receipt SHA: `2ad5202515c1890304c93198d502d681ec560f80d6e30b99b65fdabc823d76b2`.
- Bounded v3 salvage: one fixed lateral scale, one CARLA checkpoint, 10 fresh validation episodes, zero training jobs. It failed Success, Timeout and applied-lane-change gates; Cross was not run. Contract SHA: `d9b630d45d6f96b5273dd538b952e454e88d8694b7658d42451e9cf6edf35218`; completion receipt SHA: `01dfc05dfb182a18030258ce98dcbf2f95634d97c63a17809b3a40d05afb5054`.
- Bounded final summary/report and the real failed-gate launch rejection are hash-bound by `results_topo_v2_fast/bounded_final_receipt.json`, SHA `bf54f96b84a35440484249781dae60f6fd211b92fff6e4bb603eb728d0ee1e29`.
- Promotion attribution preregistration SHA: `6f92c607b89a5b7db891c565ef4904d460be3583e5b7ae2fe8cd203dd2c64e6e`.
- Formal reporting preregistration SHA: `098e1746e0f269767ac344d38e8c4b8e5caaf418bc024c5931f2e57543b21b74`; no formal run directory exists.
- Formal declaration preflight SHA: `764b3a6faa1d1b6b22a574b52a0b48734f88e5a09a6eca9a6ac8d590748fc6ae`; 120/120 unique job declarations match 2 methods × 6 scenarios × 10 seeds, 100k/50 episodes/test split, without writing a formal plan or opening test assets.
- Freeze-manifest revalidation SHA: `78fb75ceff95948bf4498a9f1975c41e455517189758a7869edfa3da7037c3da`; an authoritative in-memory rebuild matched all 301 entries exactly. The earlier missing-traffic diagnostic was corrected as a `traffic/` path-construction error and did not change frozen evidence.
- Fast promotion gate: `fail`; formal task count remains 120 and locked, with no formal directory created.
- Formal test partition accessed: **no**.
- Bounded development conclusion: **proven negative for the current v2 family and the scalar-calibration salvage**.
- Full formal result: **not produced by design because the required development gate did not pass**.
