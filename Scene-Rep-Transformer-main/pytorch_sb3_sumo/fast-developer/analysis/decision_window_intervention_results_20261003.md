# Frozen ST-RT decision-window intervention results (2026-10-03)

This report summarizes the completed `runs/strt_window_1003/recovery01` diagnostic. The machine-readable companion contains the 29 seed-by-phase paired windows and the three intervention decisions recorded for each target arm: [decision_window_intervention_results_20261003.json](decision_window_intervention_results_20261003.json). Episode traces and manifests remain the source of truth.

## Run integrity and budget

The original attempt remains preserved as aborted after 13 episodes and 3,340 controlled raw steps. Its gate failure was the seed-10004 identity replay at decision 0: the reset snapshot lacked a complete physical state for a live actor. Recovery carried that budget forward through the parent manifest SHA-256 `01168752248c5856b5d46d500ab1025ce020151efe20b7c121a8d48213dd79ec` and completed without replacing the original traces.

Recovery used 72 episodes: 12 re-recorded baselines, 2 identity replays, and 58 interventions. All 12 baselines matched the archived outcome, raw and decision lengths, and traffic variant; both identity replays passed; all 58 intervention prefixes matched. The 58 intervention episodes form 29 fixed pairs, each with 0 and 6 m/s targets applied for three policy decisions. The schedule covered 12 windows three decisions before the first internal-road state, 12 at the first internal-road pre-action state, and the first 5 terminal-proximal windows. No slot was added after observing outcomes.

The original 13/3,340 plus recovery 72/19,604 gives a cumulative 85/86 episodes and 22,944/51,600 controlled raw steps. Episode records sum to 3,163 baseline, 549 identity, and 15,892 intervention raw steps in recovery. These counters exclude reset warm-up; the exact SUMO steps inside `reset` are unavailable. The recorded raw-observer callback count is zero and initial snapshot time is 50.1 s, but neither gives the hidden warm-up tick count.

The frozen checkpoint SHA-256 is `070b05b5ad2b60ce6fc4e47ee4508f0c1d2f1dd45579a0462d50e7d6a93f6262`. Its pre-run policy-state digest is `d5a815ff8263374f91d82cf950a5358c8ac189d6f6266742c41c475aa2799d51`; the run summary reports the state fingerprint unchanged and `n_updates` 95,001 before and after. The sidecar lists 15 archived source files; all 15 exist and match their recorded SHA-256 values. The parent `recovery_linkage.jsonl` ends in `complete`, and its final recovery-manifest digest matches the manifest on disk.

## Paired outcome changes

The cases are the outcome-stratified first eight collisions and first four successes by ascending logical validation seed. Each row below counts unique seed-by-phase windows; the two target branches share their baseline and prefix, and some seeds appear in more than one phase. These counts describe the tested branches, not independent samples or overall success rates. No intervention ended in timeout.

| Anchor phase | Windows (baseline collision / success) | 0 m/s: collision → success | 0 m/s: success → collision | 6 m/s: collision → success | 6 m/s: success → collision |
|---|---:|---:|---:|---:|---:|
| Three decisions before first internal-road state | 12 (8 / 4) | 4 / 8 | 3 / 4 | 1 / 8 | 1 / 4 |
| First internal-road pre-action state | 12 (8 / 4) | 2 / 8 | 1 / 4 | 0 / 8 | 2 / 4 |
| Three decisions before baseline termination | 5 (1 / 4) | 0 / 1 | 0 / 4 | 0 / 1 | 0 / 4 |

Within the pre-entry collision cases, the paired outcomes were: both targets succeeded in 1 window, only 0 m/s succeeded in 3, neither succeeded in 4, and 6 m/s alone succeeded in none. For the four pre-entry baseline successes, both targets succeeded in 1, only 6 m/s succeeded in 2, both collided in 1, and 0 m/s alone succeeded in none. At the first internal-road state, 0 m/s alone rescued 2 of 8 baseline collisions; neither target rescued the other 6. At the terminal-proximal anchor, all four baseline successes remained successes and the one baseline collision remained a collision under both targets.

The intervention outcome totals are 24 successes and 34 collisions across the 58 branches. That is a count of repeated counterfactual variants over 29 windows, not a sample of 58 independent episodes. Per-case shaped return, raw return, reward components, episode lengths, anchor, and both arm outcomes are preserved in the JSON companion.

## How large the speed changes were

The runner replaced only the speed action coordinate and kept the policy's lateral coordinate. It used `speed_mps = clip((a0 + 1) × 5, 0, 10)` and the inverse mapping for each target. All 174 intervened action decisions changed the speed coordinate. SUMO's measured post-action speed reached 0 m/s exactly for the stop target and stayed within `2.38e-7` m/s of 6 m/s for the 6 m/s target.

The table compares each forced target with the frozen policy's current proposed target speed, mapped from its action at that same branch decision. The reference bound for one learned longitudinal residual decision is about 0.99668 m/s. Counts below are repeated action decisions (three per arm), not independent windows.

| Anchor phase | Forced target | Mean policy-proposed speed | Mean absolute override gap | Decisions with gap > 0.99668 m/s | Realized post-action speed range |
|---|---:|---:|---:|---:|---:|
| Three decisions before first internal-road state | 0 m/s | 6.668 m/s | 6.668 m/s | 26 / 36 | 0 m/s |
| Three decisions before first internal-road state | 6 m/s | 6.427 m/s | 4.246 m/s | 34 / 36 | 5.999999777–6.000000238 m/s |
| First internal-road pre-action state | 0 m/s | 4.723 m/s | 4.723 m/s | 18 / 36 | 0 m/s |
| First internal-road pre-action state | 6 m/s | 4.743 m/s | 4.713 m/s | 35 / 36 | 5.999999785–6.000000215 m/s |
| Three decisions before baseline termination | 0 m/s | 8.020 m/s | 8.020 m/s | 12 / 15 | 0 m/s |
| Three decisions before baseline termination | 6 m/s | 7.946 m/s | 4.218 m/s | 15 / 15 | 5.999999776–6.000000053 m/s |

Across all 174 decisions, 140 target overrides differed from that branch's policy proposal by more than 0.99668 m/s. The successful branches therefore show that these large, directly imposed speed targets can change some outcomes under the frozen policy's subsequent continuation. They do not show that a residual limited to roughly ±1 m/s can reproduce the target or that training will learn it. The unsuccessful branches do not establish inevitability because other speeds, timing, or continuation policies were not tested.

## History-index audit and limits

The baseline-only history audit covered 1,057 decisions from the old frozen checkpoint. It used the model's exact `_nonzero_mask` rule, `trajectory[..., 0] != 0`:

| Actor slots | Nonempty histories | Count-minus-one selecting padding | Count-minus-one differs from last valid index |
|---|---:|---:|---:|
| Ego | 1,057 / 1,057 | 0 | 0 |
| Social | 5,285 / 5,285 | 0 | 0 |

Thus, the selected baseline traces show no observed left-padding/count-minus-one exposure. This does not settle the bug's frequency in other scenes, histories, or checkpoints. Because the shared mask treats `x == 0` as padding, this audit follows the model's own validity rule rather than an independent ground-truth actor-presence label.

The terminal-proximal anchor is defined from the original baseline terminal time, so it is aligned retrospectively and is not a policy-available trigger. Prefix gates compare recorded observation/action hashes and sampled active-actor physical state, including live road/lane queries; they do not establish equality of SUMO's hidden queues or RNG state. Background traffic continues normally after a branch. Closest-approach diagnostics are geometric proxies, not true time-to-collision measurements.

The protocol snapshot archived with the run was written before recovery and still contains its pre-run status line. This result report records completion; it does not rewrite that historical protocol snapshot, the failed parent records, or the episode traces.
