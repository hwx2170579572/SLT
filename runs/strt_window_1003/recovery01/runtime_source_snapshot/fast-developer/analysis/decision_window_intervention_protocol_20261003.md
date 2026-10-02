# Frozen ST-RT decision-window interventions (2026-10-03)

Status: the first attempt is aborted at the identity-prefix gate; a bounded recovery preview is being prepared. No recovery simulation has started.

The first attempt consumed 13 controlled episodes and 3,340 ego-control raw steps: 12 baselines and one identity replay. All 12 baselines matched their archived outcome and trajectory lengths. The identity replay for seed 10004 failed at decision 0 because the reset snapshot omitted road/lane identity for a live background actor. These consumed episodes remain charged to the original 86-episode / 51,600-raw-step limit. The original run directory and traces remain unchanged.

## Hypothesis

Some observed failures may be changed by a short longitudinal intervention before entering or while clearing the intersection, while the same tested alternatives near failure may have little effect. This is a hypothesis about the existing frozen policy under specified continuations, not a claim that credit assignment is already the established training bottleneck.

## Fixed model and cases

- Scene: `intersection_sorted_depart4p0`; keep its existing vehicle, reward, collision, time-limit and traffic-seed protocol.
- Model: original 100k seed-0 `sac_mlp_d1_st_rt` final checkpoint at `runs/d0929_100k_diag/sac_mlp_d1_st_rt__intersection_sorted_depart4p0/final_model.zip`.
- Expected checkpoint SHA256: `070b05b5ad2b60ce6fc4e47ee4508f0c1d2f1dd45579a0462d50e7d6a93f6262`.
- Choose the first eight collision and first four successful cases by increasing logical evaluation seed in the previously verified 100-episode frozen control (`runs/srtact_1002/control`). No selection based on new intervention outcomes.
- This outcome-stratified subset is for mechanism diagnosis. Its success fraction cannot estimate the full evaluation population.
- CPU deterministic inference; no training, replay writes, optimizer steps or checkpoint modification.

## Prefix matching and intervention

Re-run from reset using the same logical seed. Before pairing an intervention, require exact observation and policy-action hashes and a complete, same-time physical snapshot at every pre-action decision from reset through its anchor. Capture the raw active-actor snapshot before the behavior recorder's neighbor filtering, and check both vehicle and person ID sets against read-only TraCI ID-list queries and the environment raw-step counter at each decision. Also require finite actor position/velocity/speed/heading/dimensions and current state timestamps. If the reset snapshot is absent/incomplete before the first raw callback, the runner may assemble a replacement from the environment's already-recorded actor cache, restoring diagnostic bookkeeping afterward.

For every live vehicle/person, read current `getRoadID` and `getLaneID` values through TraCI without calling `simulationStep`. Require simulator time, raw-step counter, and both live ID sets to remain unchanged across the reads. These live values are authoritative for the current physical snapshot. Preserve each pre-query cache value and its `context_sample_raw_step`: an explicitly older cache value may differ and is recorded as stale, while a nonempty same-raw-step disagreement, future cache timestamp, getter failure, missing/extra actor, or time/step/ID instability makes the snapshot incomplete. Empty or missing live road/lane values remain incomplete; they are never accepted by comparing two missing values. Do not query actor routes or hidden future routes. At a mismatch, retain a compact per-actor field diff; baseline rows retain physical snapshots and intervention rows retain their anchor snapshot. Do not treat a model checkpoint as a saved simulator state.

At each policy decision, the runner also records a read-only audit of `observation["trajectory"]` using the same `_nonzero_mask` rule as the encoder (`trajectory[..., 0] != 0`). For each ego/social actor slot it records valid-frame count, the count-minus-one index, the highest valid index, whether count-minus-one selects a mask-false slot, and their signed index difference. This is calculated after policy prediction, does not modify the observation or action, and is excluded from prefix hashes. It measures the behavior of the frozen checkpoint under the model's existing mask; because x=0 is treated as padding by `_nonzero_mask`, it is not an independent ground-truth padding label. The actor slots are ego slot 0 and social slots 1 onward; policy observations do not expose actor IDs.

These checks establish equality of the recorded observations/actions and active vehicles' sampled physical state, not equality of SUMO's full internal state, pending departure queues, or hidden RNG state. Identity replays remain a required gate. Any incomplete or mismatched physical prefix is NA/rejected rather than accepted because both hashes are missing.

In recovery, re-record baselines for all twelve selected cases and require each to reproduce the archived outcome and trajectory length. Then run two independent identity replays (one collision, one success) and require matched observations/actions and terminal outcome before proceeding. The per-decision history-index audit and a twelve-baseline exposure summary are saved with these re-recorded baselines. The selected cases remain outcome-stratified and do not estimate exposure in the full evaluation or training distribution; an audit on this older checkpoint does not establish the behavior of a later checkpoint with an index fix.

Predefine anchors from the newly reproduced baseline:

1. Three decisions before first entry onto an internal junction road.
2. First decision whose pre-action ego road is internal.
3. Three decisions before baseline termination.

Clamp only when there is an existing valid decision; mark unavailable phases NA and merge identical decision indices. Report realized phase and time to the original terminal event, rather than assuming every anchor is a distinct behavioral stage.

Reserve intervention slots before running any intervention. Order them by phase (pre-entry, first internal-road decision, terminal-proximal), then ascending selected seed. Under the recovery cap, the fixed schedule reserves all twelve pre-entry slots, all twelve first-internal slots, and the first five terminal-proximal slots. Keep the two target speeds (0 and 6 m/s) together as a pair in every reserved slot. Missing anchors, incomplete prefixes, or merged phases consume their reserved slot and do not shift later terminal slots forward.

At each eligible reserved anchor, separately request target speed 0 or 6 m/s for at most three normal policy decisions (about 0.9 seconds at repeat=3). Use the environment's normal action interface: `a_speed = target_speed/5 - 1`. Keep the policy's lateral action. After the intervention return to the frozen policy. End immediately if the environment terminates.

The policy continues to predict from the actual changed state. Background vehicles respond normally. Do not impose their recorded trajectories after branching. A same-seed replay and matched prefix do not guarantee identical future random-number consumption after trajectories diverge.

Reject mismatched prefixes from paired interpretation. Stop and investigate failed identity checks rather than extending the budget or silently loosening tolerances. Record action changes, requested and realized speed, and whether the intervention actually changed execution. A candidate identical to the policy's action is not evidence that a window is ineffective.

## Hard budget

- Original total cap: at most 86 controlled episodes and 51,600 ego-control raw steps (600 per episode). The failed first attempt has already consumed 13 episodes / 3,340 raw steps.
- Recovery must use a new `recovery01` directory and validate the aborted parent manifest, source/checkpoint hashes, and consumed-budget records. It re-runs 12 baselines and 2 identity replays, leaving at most 58 intervention episodes (29 fixed pairs). The worst-case cumulative episode count is 13 + 12 + 2 + 58 = 85, leaving one episode unused; the worst-case cumulative raw-step count is 3,340 + 72 x 600 = 46,540. Any fixed-slot gaps reduce actual runs; they do not permit outcome-adaptive additions.
- Normal reset/warmup simulation is separate overhead. The current environment advances SUMO inside `reset` and resets its local raw counter without exposing that tick count; therefore exact reset/warmup raw steps are recorded as unavailable, with raw-observer callback count and initial snapshot simulation time kept as separate fields. Do not infer warmup ticks from a zero callback count or call the ego-control bound a bound on all simulator ticks.
- No adaptive additional cases, action sweeps or training. Partial failures and unused budget remain visible in the manifest.

## Outputs and interpretation

For each case/anchor/action save checkpoint/source identity, seed, prefix checks, baseline/proposed/executed action, actual speed, route/lane state, outcome, raw/decision steps, collision evidence, shaped/raw return and reward components. Summarize paired collision-to-success, success-to-collision, timeout changes, phase coverage and intervention execution strength.

These diagnostic speed changes are larger than the new learned residual's approximately 1 m/s bound. A successful intervention establishes an opportunity for the tested action/continuation, not that the new bounded residual can reproduce it. An unsuccessful intervention does not prove inevitability: untested actions, timing or subsequent policies could differ.

Use findings to decide whether later work should investigate earlier decisions, intersection clearance, execution constraints, value calibration or credit assignment. Keep implementation/optimization/data/evaluation explanations separate from this behavioral evidence.
