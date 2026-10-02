# Frozen STRT RouteAct check — 2026-10-02

## Scope and identity

This is a read-only summary of the completed two-arm frozen-policy evaluation in `runs/srtact_1002`. Both arms loaded the same `sac_mlp_d1_st_rt` final checkpoint (`runs/d0929_100k_diag/sac_mlp_d1_st_rt__intersection_sorted_depart4p0/final_model.zip`, SHA-256 `070b05b5ad2b60ce6fc4e47ee4508f0c1d2f1dd45579a0462d50e7d6a93f6262`). The control used the original evaluator; RouteAct wrapped only environment lane-action forwarding. Neither arm trained or resumed: evaluation ran on CPU, `learning_or_replay_updates=0`, and `_n_updates` stayed at 95,001. The state-dict fingerprint was identical before and after evaluation in both arms.

Both arms used `intersection_sorted`, `depart_scale=4.0`, the validation split, and logical episode seeds 10000–10099. Each arm completed 100 episodes. Their seed and `traffic_variant` matched episode by episode; 30 route templates were reused from the same sorted traffic pool. The old final evaluation also matched the new control on all 100 seeds and template names. This is a paired evaluation over the fixed checkpoint and this reused pool, not 100 independent traffic layouts or multiple training seeds.

## Results

| Evaluation | Success | Collision | Timeout | Off-route | Mean shaped return (std) | Mean raw return | Mean raw / decision steps |
|---|---:|---:|---:|---:|---:|---:|
| Old STRT final evaluation | 63/100 | 37/100 | 0/100 | 0/100 | Not reported on the new shaped-reward protocol | 0.260000* | 268.99 / 90.03 |
| Frozen control, current evaluator | 63/100 | 37/100 | 0/100 | 0/100 | 4.389848 (10.389009) | 0.260000 | 268.99 / 90.03 |
| Frozen RouteAct wrapper | 63/100 | 37/100 | 0/100 | 0/100 | 4.389848 (10.389009) | 0.260000 | 268.99 / 90.03 |

`*` The old artifact stores `episode_return` rather than a separate `raw_episode_return`; its per-episode values match the current control’s raw return exactly for all 100 episodes. The current evaluator’s `episode_return` is shaped (`environment_step_reward_v2`), while `raw_episode_return` is separately recorded. These return columns therefore must not be compared as if they had the same definition.

The control-to-RouteAct 3×3 outcome transition matrix contains only `success→success: 63` and `collision→collision: 37`; there were no outcome changes. The old-evaluation-to-current-control pairing has the same matrix. For old-to-control, seed, template, terminal outcome, raw steps, decision steps, and per-episode raw return match in all 100 episodes. For control-to-RouteAct, terminal outcome, raw and decision steps, shaped return, and raw return match in all 100 episodes.

For both current-evaluator arms, the six mean reward components were: success `+6.300000`, collision `−3.700000`, off-route `0`, timeout `0`, step cost `−0.900300`, and progress `+2.690148`. Their sum matches mean shaped return `4.389848`; the maximum per-episode component-reconciliation error was `3.55e−15`, with 100/100 component coverage. Raw mean return was `0.26` (raw std `0.965609`). These raw and shaped values are recorded separately in the paired JSON.

## What this checks—and what it does not

The RouteAct sidecar has 9,003 decision rows. It recorded zero vetoes, zero action differences, and zero decisions with a missing reason. Reasons were `target_lane_out_of_range` 5,662, `internal_road` 1,790, `no_next_edge` 1,513, and `hold_command` 38. Thus the wrapper did not alter any action in these 100 episodes. Equal outcomes establish that the wrapper was behavior-neutral on this evaluation, not that its intervention improves or harms driving. In particular, this run does not test whether a veto can prevent route-incompatible lane requests; it also does not show that such a constraint resolves STRT’s 37 collision outcomes.

A fresh RouteAct training run could test the narrower hypothesis that learning adapts to the wrapper during exploration and replay. That would be a separate experiment: the present frozen result gives no evidence that training adaptation will improve success or collision rate. Keep the intervention claim limited to the condition actually exercised, and report veto coverage alongside any later performance result.

## Source artifacts

- Paired per-seed summary: `runs/srtact_1002/routeact_paired_summary.json`.
- Current control result and behavior summary: `runs/srtact_1002/control/evaluation_results.json` and `runs/srtact_1002/control/diagnostics/eval/summary.json`.
- Current RouteAct result and behavior summary: `runs/srtact_1002/routeact/evaluation_results.json` and `runs/srtact_1002/routeact/diagnostics/eval/summary.json`.
- RouteAct decisions: `runs/srtact_1002/routeact/diagnostics/eval/route_action_consistency.jsonl`.
- Previous STRT result: `runs/d0929_100k_diag/sac_mlp_d1_st_rt__intersection_sorted_depart4p0/evaluation_results.json`.
- Frozen evaluation identity, completion status, commands, and source copies: `runs/srtact_1002/experiment_manifest.json`, each arm’s `worker_status.json` and `command.json`, and `runs/srtact_1002/source_archive/manifest.json`.

The launch commands and archived runner, D1 evaluator, and route-action wrapper point to the `Scene-Rep-Transformer-main1` workspace. Their current source hashes match the archived hashes. The result remains limited to one training seed and this fixed evaluation protocol.
