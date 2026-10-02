# Topo and three-slot incremental study

## User-confirmed scope (2026-09-29)

Two independent CUDA workers will train from scratch for a continuous 100,000
raw simulator steps each, with training seed 0 and the existing matched
100-episode evaluation protocol:

1. `sac_mlp_d1_st_rt_topo`: add Topo to the recorded ST-RT stage.
2. `sac_mlp_d1_st_rt_topo_3slot`: add only the ego/social/route structure to
   that Topo stage. This run also starts from scratch; it does not resume
   the first run's checkpoint.

The three slots have dimensions 32/64/32 and feed the same 128-dimensional
SAC feature interface. Graph-SLT and soft slot balance loss (SBS) remain
disabled in the new three-slot experiment. Existing `full` keeps its
historical implementation and auxiliary losses.

The purpose is to identify the effects and possible failure causes of
individual parts of `full`. SAC+MLP is the pure RL baseline. MST+SLT is the
strong comparator for the complete method, not a required threshold at
each incremental stage. This stage uses one training seed by user choice.

## Comparisons and decisions

- Compare Topo with the recorded ST-RT run at
  `runs/d0929_100k_diag/sac_mlp_d1_st_rt__intersection_sorted_depart4p0`
  (63 success, 37 collision, 0 timeout in the recorded evaluation).
- Compare three-slot with the new Topo run, preserving the same training
  budget, traffic density, evaluation seed/traffic pairs and final
  checkpoint rule.
- Save the full paired outcome matrix, including collision-to-success,
  success-to-collision and every transition involving timeout. A net gain
  in success is not sufficient evidence if timeout increases or matched
  cases are absent. Seed/traffic pairs must match before interpretation.
- Use normal training/evaluation diagnostics to establish that Topo was
  active and received finite nonzero critic gradients, then inspect slot
  scale, variation, relation use and parameter-group coverage.
- Equal or worse results trigger analysis of the implicated module. Do
  not stack another module before understanding that result.

These runs provide evidence for the specified seed and protocol; they do
not establish robustness across training seeds or held-out traffic.

## Diagnostics collected during the existing computation

The existing behavior streams retain policy actions, applied actions,
actual speeds, nearby/conflicting vehicles, TTC validity and collision
locations. New representation streams attach activation snapshots to
existing encoder forwards and gradient snapshots to the existing critic
backward before gradient clipping. No extra rollout, prediction or
backward is required to collect these observations.

- Topo: active-path flag, lane compatibility/attention coverage, relation
  availability and use, fusion scale, and parameter-group gradients.
- Three-slot: enabled flag and dimensions, per-slot magnitude/energy,
  variation with its valid sample count, fusion scale and per-slot
  parameter gradients. Unequal-dimensional slots are not compared with
  an undefined cosine similarity.
- Gradient observations distinguish absent gradients, allocated zero
  gradients and nonfinite gradients. Group norms can overlap and must
  not be added together.
- Statistics are grouped by source. Replay-batch training diagnostics are
  not assigned to the current environment episode. Evaluation snapshots
  are collected after the existing prediction and before the associated
  environment step.
- Missing or nonfinite scalars are null with coverage counts. Batch size
  one cannot establish cross-sample variance or absence of collapse.

Activation magnitude and attention allocation describe computation;
they are not by themselves causal evidence that a component improved
the policy. That attribution comes from the matched stage comparisons.

## Execution status

**2026-09-30 recovery update:** the original three-slot worker later exited
after Windows denied a diagnostic-summary replacement. The launch snapshots
below are historical and do not indicate that both original workers remain
healthy. See [the I/O repair and restart record](<D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/Scene-Rep-Transformer-main/pytorch_sb3_sumo/fast-developer/analysis/topo_3slot_io_recovery.md>)
for the confirmed failure, regression tests and replacement-run status.

Implementation verification completed on 2026-09-30. Five focused encoder
tests and seven existing V2 tests passed. The core behavior, gradient-event
and paired-comparison suite also passed; the evaluation-budget guard has
an additional targeted test.

The normal CUDA smoke run at
`D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/t0930_topo3_smoke_01`
completed for both workers with exit code 0: each trained for 300 raw
steps, performed 241 updates after its smoke-only 60-step warmup, and
evaluated two episodes. Both training and evaluation diagnostic error
counts were zero. Each training stream contained one critic-gradient
sample and five activation samples. Topo core gradients were nonzero;
the three-slot run also had nonzero gradients in all three slot
projections, with no nonfinite gradient elements. Evaluation produced
400 and 230 representation rows respectively, exactly matching the
decision counts, with unique sample indices and zero snapshot age.
These smoke outcomes are implementation checks, not research results.

The formal hidden supervisor started at **2026-09-30 00:31:15 Asia/Shanghai**,
PID **56072**, under
`D:/Program Files (x86)/paper/Scene-Rep-Transformer-main1/runs/t0930_0023_topo3_100k`.
The exact command and source snapshot are preserved in the run root;
`launcher_supervisor.json` identifies the supervisor and
`launcher_status.json` is the current worker-state source of truth.
This entry records launch, not completion or a performance result.

The command selects exactly `sac_mlp_d1_st_rt_topo` and
`sac_mlp_d1_st_rt_topo_3slot`, sets `--max-steps 100000`,
`--depart-scale 4.0`, `--checkpoint-frequency 10000`,
`--behavior-diagnostics`, and the previous `runs/d0929_100k_diag`
reference root. It contains no smoke or resume option. Training seed
is 0, learning rate is 1e-4, warmup is 5,000 raw steps and final
evaluation is 100 episodes per method. Checkpoint frequency is
10,000 **raw** steps: the callback reads `_raw_steps_seen`.
Each model uses one continuous `learn` call with a 100,000-raw-step
budget. Model training/optimizer cadence is unchanged by checkpoint
saving.

Initial launch verification on 2026-09-30 confirmed both workers were
running and advancing, each reaching at least **2,392 raw steps** with
zero updates while still inside the 5,000-step warmup:

| Method | Worker PID | Device | Initial state |
| --- | --- | --- | --- |
| `sac_mlp_d1_st_rt_topo` | 17496 | CUDA | Training, warmup |
| `sac_mlp_d1_st_rt_topo_3slot` | 9240 | CUDA | Training, warmup |

Both worker PIDs also appeared in the GPU compute-process listing.
The per-worker `arguments.json` files confirm seed 0, 100,000 raw
steps, 5,000-step warmup and `smoke=false`. Consult their `progress.json`
files for current progress; the numbers above are launch snapshots.

Run-root provenance files are `experiment_manifest.prelaunch.json`,
`launcher_supervisor.json` and `source_snapshot.json`. After completion,
the runner writes `comparison_strt_to_topo.json` and
`comparison_topo_to_3slot.json`, containing matched seed/traffic outcome
transitions and pairing/episode-budget checks. Each worker keeps the
existing behavior streams and the added `diagnostics/train/representation.jsonl`
and `diagnostics/eval/representation.jsonl`, with phase summaries.

The `train_replay` encoder sampling-source label denotes the configured
training phase. Because the online extractor is shared with rollout
prediction, this label alone does not establish that every activation
snapshot came from replay. Use the event source, batch size, gradient
mode, sample index and age together. Replay/training activation events
are not assigned to rollout episodes. Critic gradient events are sampled
specifically from the TD backward and have an exact update index.
