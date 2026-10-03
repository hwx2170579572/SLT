# Independent scene-event representation route

This package and the new `train_scene_event_v1.py` / `validate_scene_event_v1.py` entry points are separate from the historical D1/full/MST+SLT implementation. The current build implements M0/M1 infrastructure; later learned-progress/shared-mode stages require their own evidence gates. See `../analysis/scene_representation_redesign_20261003/experiment_protocol.md` and `stage_acceptance.md` for authoritative status. A module existing on disk does not mean its stage has passed.

## Ownership and interfaces

- `schema.py`, `collector.py`, `env.py`: current and past observable state, explicit masks, remaining task time, ordinary raw-tick capture. Tracking keys remain outside policy observations.
- `geometry.py`, `map_cache.py`: public static lane connectivity, movement candidates and conflict geometry. No social future-route oracle.
- `temporal.py`, `polyline.py`, `dual_graph.py`, `encoder.py`: generic M0 scene encoder with 128-dimensional output.
- `event_projection.py`, `event_graph.py`: M1 deterministic passage-event representation and route-conditioned cross-actor messages.
- `protocol.py`, `replay.py`, `policy.py`: explicit task/discount conventions, actual-k n-step assembly, immutable raw-observation replay, and independent online/target encoder SAC. Critics receive `(z, action)`. M0 and M1 share a 10000-transition uniform-sampled/FIFO replay capacity, selected for host-memory limits rather than performance tuning; the Z=90 dynamic observation arrays total 221140 bytes each, and the two-worker one-decision-terminal array bound is 8.85 GB before Python, model, and environment overhead.
- `future_targets.py`: delayed factual targets produced from subsequent ordinary observations. This is data collection, not a learned predictor or a counterfactual model.
- `diagnostics.py`, `provenance.py`, `trainer.py`, `acceptance.py`: bounded diagnostics, independent result roots, identities, reward accounting and implementation gates.

All trainable models use actor inputs detached from the shared encoder initially. Encoder/critic parameters have one optimizer owner; target parameters receive synchronization only. The new finite task deadline is 60 seconds and is a true terminal event. Intermediate-entropy and off-policy corrections are not silently added to the retained approximate reward-only multistep target; that limitation is explicit in the experiment protocol.

## Running checks

Use the project interpreter `D:\Programs\Anaconda\envs\pytorch\python.exe`, with the working directory set to `fast-developer`.

```powershell
& 'D:\Programs\Anaconda\envs\pytorch\python.exe' train_scene_event_v1.py --mode check --method sac_scene_dualgraph_v1
```

`check` prints the configuration only and starts no simulator. `smoke` uses its own declared short budget (default 1200 raw steps, learning starts 128, two evaluation episodes); it must use a new root. It cannot contribute to a formal 100k result table.

```powershell
& 'D:\Programs\Anaconda\envs\pytorch\python.exe' train_scene_event_v1.py --mode smoke --method sac_scene_dualgraph_v1 --run-root '<new M0 smoke root>'
& 'D:\Programs\Anaconda\envs\pytorch\python.exe' train_scene_event_v1.py --mode smoke --method sac_scene_eventgraph_cv_v1 --run-root '<new M1 smoke root>'
```

`validate_scene_event_deadline_v1.py --run-root ...` performs one separate, fixed-action 600-raw contract episode to verify physical clock, remaining time, terminal flags and reward accounting. It trains no model and is not a policy evaluation result.

After the implementation review and both smokes, `validate_scene_event_v1.py --m0-smoke ... --m1-smoke ... --deadline-contract ... --output ...` runs the independent tests and checks source/checkpoint/reward/diagnostic identities. A stale receipt is rejected. Formal `--mode train` requires this receipt and exactly 100000 raw steps with 100 final evaluation episodes. Passing these engineering checks is not evidence of a performance advantage, calibration, or joint-mode usefulness.

## Diagnostic units and data separation

`episodes.jsonl` is the episode table; a final collection cutoff is recorded as incomplete and is not a task failure. `actions.jsonl` and `observation_audit.jsonl` have real decision rows. Shadow variants share a bounded set of source states and never increase simulator episodes. Six reward fields are episode cumulative totals, so the collector reconciles their **increments** against each returned reward. The inherited raw environment return is logged in a separate column.

`factual_train/` and `factual_eval/` contain separate compressed shards. Future validity masks distinguish an observed coordinate of zero from a missing target; later observed samples never replace the anchor input. End-of-episode censoring retains already observed samples. Tracking keys are audit-only label-join metadata and must not enter a model input. These targets describe the collecting policy's factual future and do not predict a response to an unexecuted ego action. Split datasets by episode and retain policy-stage/censor coverage when comparing calibration.

Public map hashes, source ZIP/member hashes, checkpoint hashes, evaluation seeds, executed raw/decision/update counts and actual device are retained in each root. Failed smoke roots remain available for diagnosis; do not delete or reuse them to hide failed attempts.
