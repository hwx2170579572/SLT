# Geometric collision evidence logging and test audit (2026-10-03)

## Passive logger change

`envs/sumo/sumo_env.py` now retains the identity of the first vehicle or pedestrian already found by the existing oriented-box collision fallback. The environment still calls the same `_geometric_collision() -> bool` hook from `_events_after_step`; the original short-circuit order and collision decision are unchanged. `_events_after_step` clears the metadata immediately before that call, so a subclass or test override that returns only a boolean cannot leak a previous hit. The cache is also cleared on reset and close.

When behavior snapshots are enabled, `geometric_collision_evidence` joins that first-hit ID against `current_payloads` already collected for the same raw-step snapshot. It reports the capture status, collision/snapshot times, whether the actor appeared in the last policy observation, and `observed_neighbor_raw_step`. A missing actor or time mismatch is recorded as unknown. The logger does not make another TraCI call, enumerate partners again, or change action, reward, observation, or terminal flags. “First hit” is the detector's first overlap, not a complete list of overlapping actors, a SUMO-reported collision participant, or a fault determination.

The change is in [`sumo_env.py`](../../envs/sumo/sumo_env.py), principally cache lifecycle and `_events_after_step` around lines 353, 434, 1602–1605, and 2547; the first-hit branches are around 1641–1699, and snapshot matching is around 1235–1310. The offline regression tests are in [`test_sumo_geometric_collision_evidence.py`](../../tests_sb3_sumo/test_sumo_geometric_collision_evidence.py). The separate DARRL terminal-resolution fixture in [`test_random_intersection_darrl_collision.py`](../../tests_sb3_sumo/test_random_intersection_darrl_collision.py) was updated to initialize `_behavior_lane_apply_diagnostics`, matching the production constructor when the test builds an env via `__new__`.

## Explicit pure-test verification

Working directory: `D:\Program Files (x86)\paper\Scene-Rep-Transformer-main1\Scene-Rep-Transformer-main\pytorch_sb3_sumo`.

```powershell
& 'D:\Programs\Anaconda\envs\pytorch\python.exe' -m pytest -p no:cacheprovider -q tests_sb3_sumo/test_sumo_geometric_collision_evidence.py tests_sb3_sumo/test_random_intersection_darrl_collision.py
```

Result: **15 passed in 0.36s**. This explicit command names only the mocked/offline collision-evidence and terminal-resolution tests; it does not include `test_sumo_env.py` or start SUMO. The related compile check also exited 0:

```powershell
& 'D:\Programs\Anaconda\envs\pytorch\python.exe' -m py_compile envs/sumo/sumo_env.py tests_sb3_sumo/test_sumo_geometric_collision_evidence.py tests_sb3_sumo/test_random_intersection_darrl_collision.py
```

An earlier full run of the two pure test files reported 13 passed and one failed because the `__new__` fixture lacked `_behavior_lane_apply_diagnostics`; after adding the constructor-equivalent fixture field, the complete explicit command above passed all 15. No production behavior was changed to accommodate that fixture.

## Accidental combined pytest invocation

One earlier command unintentionally included the real-environment test module:

```powershell
& 'D:\Programs\Anaconda\envs\pytorch\python.exe' -m pytest -p no:cacheprovider -q tests_sb3_sumo/test_sumo_geometric_collision_evidence.py tests_sb3_sumo/test_sumo_env.py tests_sb3_sumo/test_random_intersection_darrl_collision.py
```

It ran from the working directory above. The tool returned 21 progress dots after about 30.6 seconds, but the wrapper did not retain the running process's session ID or final pytest summary/exit code. Therefore the exact completed item count, total simulator resets/steps, and eventual command exit status are **unknown**. Under the explicit argument order and standard pytest ordering, the 21 reported passes correspond to the 9 collision-evidence tests followed by the first 12 cases in `test_sumo_env.py`: its four pure checks, six parameterized real-environment reset/one-step cases, and the first two parameterized complete-episode cases. Thus there is evidence that at least six reset/one-step cases and two episode cases ran; their exact raw simulation steps are not recoverable. Further test execution after that output is unknown. A later process check found no running Python or SUMO process; an attempted Windows command-line query was access-denied, so it does not identify the earlier pytest child.

The included `test_sumo_env.py` cases use the six `base_runnable_scenarios()` (`left_turn`, `cross`, `carla`, `roundabout`, `roundabout_easy`, `roundabout_medium`), not `intersection_sorted` or either `sortct_*` root. See `scenario_registry.py:40` for `config_path` (under `envs/sumo/scenarios/<scenario>/scenario.sumocfg`) and `:139` for the base-scenario selector; the real reset/step cases are `test_sumo_env.py:64-118`, with the checker/cross/legacy cases at `:121-149`. They instantiate `SumoSceneEnv(scenario=...)` without a `run_dir` or explicit `sumo_args`. `sumo_env.py:387-406` constructs SUMO with the scenario config, seed, `--no-step-log`, `--duration-log.disable`, `--quit-on-end`, and `--time-to-teleport -1`; `:418` starts TraCI with that command. It does not pass an output directory or tripinfo/FCD/route-output option. A search of those six referenced configs found no `output`, `tripinfo`, `fcd`, or `vehroute` output setting. The two collision-specific test files use mocked APIs or command-only env objects; they do not start SUMO. Thus the reviewed test code has no configured write path to `runs/sortct_1002`, `runs/sortct_frozen_1003`, or a test temporary directory. This is source/config-path evidence, not a filesystem-wide proof that the simulator emitted no incidental runtime files; run-root mtimes were not audited.

That accidental pytest invocation was separate from `runs/sortct_frozen_1003`: it is not one of the two control gates or three 40-episode intervention arms, and it must not be reported as part of the frozen 122-episode diagnostic or as a performance result. It did involve real SUMO tests, so it must not be described as “no extra simulator execution.”
