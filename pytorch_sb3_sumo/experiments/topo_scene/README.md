# Topology-Temporal Graph-SLT experiment package

This package is the executable evidence contract for the L1 method. It does
not contain assumed improvements: every number in a summary is parsed from a
completed `train_paper_sb3_sumo.py` run whose implementation ID, raw-step
clock, checkpoint audit, final evaluation, diagnostics, and performance
profile all match the contract.

## Claim–evidence matrix

| Claim | Reviewer question | Evidence needed | Workload | Baselines | Metrics | Result | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1: Full improves closed-loop driving without a safety regression | Does the method solve the stated control problem? | Matched final-checkpoint comparison | left_turn, cross, roundabout_medium | scene_rep | success, collision, return | Full 0.60/0.40/0.20 vs MST+SLT 1.00/0.00/1.00 on the completed screen | rejected_in_development |
| C2: Per-frame vehicle interaction is useful | Is early temporal pooling the bottleneck? | `temporal_graph` mechanism ablation and risk probes | same replay/evaluation traffic | scene_rep | success, collision, TTC/min-distance probe | TemporalGraph is safe but does not exceed the ceiling baseline or its risk probes | not_supported_as_an_improvement |
| C3: Road topology adds value | Does Full beat the temporal graph without topology? | Full vs `temporal_graph`, attention locality, route probe | three scenarios | temporal_graph | success, collision, route probe, attention distance | Original Full is worse; BalancedSlots repairs stability but adds no primary-metric gain | rejected_for_original_full |
| C4: Overhead is bounded | Is graph computation operationally acceptable? | Matched profiling | left_turn plus confirmation workloads | scene_rep, temporal_graph | parameters, train ms/update, inference ms, peak GPU MB | All four engineering metrics are measured; no acceptance threshold was predeclared | measured_without_acceptance_threshold |

## Execution order

```powershell
conda run -n llm_pipeline python tools/run_topo_experiments.py validate

conda run -n llm_pipeline python tools/run_topo_experiments.py run `
  --phase pilot --device cuda

conda run -n llm_pipeline python tools/run_topo_experiments.py summarize `
  --phase pilot

conda run -n llm_pipeline python tools/run_topo_experiments.py run `
  --phase development --device cuda

conda run -n llm_pipeline python tools/run_topo_experiments.py summarize `
  --phase development

conda run -n llm_pipeline python tools/run_topo_experiments.py run `
  --phase diagnostic --device cuda

conda run -n llm_pipeline python tools/run_topo_experiments.py summarize `
  --phase diagnostic

conda run -n llm_pipeline python tools/run_topo_experiments.py run `
  --phase iteration --device cuda

conda run -n llm_pipeline python tools/run_topo_experiments.py summarize `
  --phase iteration
```

The confirmation command is deliberately gate-protected. It refuses to start
unless `results_topo_scene/development/gate_decision.json` records
`decision: pass`.

After all development/diagnostic/iteration checkpoints exist, rebuild the
primary learning curves with one explicit paired seed sequence. Historical
`evaluations.npz` files from runs created before the callback seed-reset fix
remain exploratory only.

```powershell
conda run -n llm_pipeline python tools/reevaluate_checkpoint_curve.py `
  --run-dir results_topo_scene/development/runs/development__scene_rep__left_turn__seed0 `
  --run-dir results_topo_scene/development/runs/development__topo_scene__left_turn__seed0 `
  --run-dir results_topo_scene/diagnostic/runs/diagnostic__temporal_graph__left_turn__seed0 `
  --run-dir results_topo_scene/iteration/runs/iteration__topo_scene_balanced__left_turn__seed0 `
  --episodes 20 --seed-start 10000 --device cuda `
  --output results_topo_scene/diagnosis/fixed_checkpoint_curves.json
```

## Development gate

The gate is a screening decision, not a paper conclusion. It requires:

- all control and Graph-SLT diagnostics finite;
- `Full - MST+SLT` success at least `+0.05`, or return strictly positive;
- collision-rate increase no more than `0.02`;
- topology top-1 attention-distance p90 no more than `40 m`;
- mean topology-attention mass within `40 m` at least `0.5`.

A finite run that misses one of these becomes `diagnose`. Non-finite values,
graph overflow, or corrupt checkpoints become `stop`. A `diagnose` outcome
routes to the `temporal_graph` ablation and representation probes; it does not
authorize a larger network.

The real development outcome was `diagnose`, so the confirmation matrix was
not started. The permitted single-variable iteration added independent
non-affine LayerNorm only after the three structured slots. It removed the
40k/50k collision collapse but did not create a primary-metric gain over the
MST+SLT or TemporalGraph ceilings; the final decision is
`retain_temporal_only` within the graph variants, while MST+SLT remains the
empirical deployment reference.

## Result table

| Scenario | Method | Success ↑ | Collision ↓ | Return ↑ | Off-route ↓ | Completion time ↓ | Train ms/update |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| left_turn | MST+SLT | 1.000 | 0.000 | 1.000 | 0.000 | 12.335 | 103.539 |
| left_turn | TemporalGraph+Graph-SLT | 1.000 | 0.000 | 1.000 | 0.000 | 16.580 | 143.832 |
| left_turn | Full | 0.600 | 0.400 | 0.200 | 0.000 | 10.250 | 272.519 |
| left_turn | Full+BalancedSlots | 1.000 | 0.000 | 1.000 | 0.000 | 18.390 | 198.093 |
| cross | all methods | gate-blocked | gate-blocked | gate-blocked | gate-blocked | gate-blocked | gate-blocked |
| roundabout_medium | all methods | gate-blocked | gate-blocked | gate-blocked | gate-blocked | gate-blocked | gate-blocked |

All numeric rows above come from completed 50k raw-step, seed-0 runs and the
same deterministic evaluation seeds 10000..10019. Completion time is computed
over successful episodes, so the Full value must not be read as compensating
for its 40% collision rate. Cross-scenario and between-training-seed evidence
was not generated after the frozen gate failed.
