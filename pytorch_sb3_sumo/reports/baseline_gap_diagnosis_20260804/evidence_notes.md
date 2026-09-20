# Evidence and QA notes

## Reporting job

- Question: explain why migrated PyTorch + Stable-Baselines3 + SUMO SAC/PPO results appear far from the paper.
- Audience: technical.
- Comparison basis: paper Table I (50 test episodes), released TensorFlow/SMARTS source behavior, v14 result artifacts, and controlled 50-episode re-evaluations of existing checkpoints.
- Decision supported: determine whether retraining is needed, which existing artifacts are valid, and which protocol must be used before continuing the 5 x 2 x 5 matrix.

## Required-structure mapping

- Title: first markdown block.
- Technical summary: second markdown block.
- Key findings with visual evidence: SAC reconciliation chart, matrix-quality table, PPO seed table, and adjacent interpretation blocks.
- Scope, data, and metric definitions: metric/protocol block.
- Methodology: controlled-check block.
- Limitations, uncertainty, and robustness: limitations block.
- Recommended next steps: remediation block.
- Further questions: final block.

## Source inventory

- Paper PDF: `tmp/pdfs/scene_rep_transformer_arxiv_v3.pdf`.
- Released test runner: `tools/test.py`.
- Migrated final-evaluation implementation: `pytorch_sb3_sumo/tools/train_sb3.py`.
- v14 protocol: `pytorch_sb3_sumo/experiments/sb3_sumo_paper/protocol_baselines_v14.json`.
- Orchestrator state: `results_sb3_sumo_paper/paper_baselines_sac_ppo_v14_frozen/orchestrator_state.json`.
- Per-job evaluation JSON and controlled diagnostic JSON under the same result root.
- Reproducible aggregation: `reproduce_diagnosis.ps1` in this report directory.

## Data-quality classification

- Planned grain: one job per algorithm x scenario x training seed; 50 evaluation episodes per job.
- The orchestrator reports 9 complete, 2 partial, and 39 pending jobs.
- Only six of the nine completed artifacts contain non-null released traffic-variant provenance. Three latest SAC artifacts used the generic environment at final evaluation and are invalid for paper comparison.
- The two partial PPO jobs finished training/checkpoint creation but crashed during final evaluation because the generic environment returned a dictionary observation to an RGB Box policy.

## Visual contract and chart map

- Section: SAC result reconciliation.
- Analytical question: does the 0% result describe the learned policy or a broken evaluation path?
- Takeaway: the same checkpoints recover to 90%/84% under the intended frozen paper environment and 74%/76% under source-all; the stored 0% values are invalid measurements.
- Family/type: comparison, horizontal `bar`.
- Data: eight discrete 50-episode rows; no temporal inference.
- Encodings: success rate on the quantitative x-axis, scenario/protocol label on the nominal y-axis; no redundant color grouping.
- Palette: single-root preferred, with validity encoded in labels so the chart remains interpretable without color.
- Footprint: full-width HTML report block.
- QA surface: packaged portable report at desktop and narrow widths.
- Omitted chart: PPO has only three completed seed observations. Exact lookup and grain warnings matter more than shape, so a table is safer than a second chart.

## Method and robustness checks

1. Reproduced the current stored summaries and classified environment provenance via `traffic_variant`.
2. Re-evaluated the same SAC checkpoints for 50 episodes in the intended paper environment under both `frozen_80_20` and `source_all`, holding model, seeds, action repeat, control proxy, episode limit, and deterministic inference fixed.
3. Re-evaluated three PPO left-turn checkpoints for 50 episodes under `source_all` and computed the equal-weight seed mean.
4. Compared results with paper Table I and inspected the original and migrated evaluation code paths.

## Caveats and omitted claims

- The paper does not release its exact held-out traffic assets, policy-selection implementation, or a runnable plain-SAC entry point. Exact bitwise reproduction is impossible from the release alone.
- A single seed is not used to infer mean algorithm quality.
- The report does not claim that `source_all` is the unpublished paper test set; it is the executable released-source traffic cycle.
- No performance claim is made for Double Merging, Roundabout-B, or Roundabout-C until their correct final evaluations exist.
- The result matrix is incomplete, so no overall five-scenario score is reported.
