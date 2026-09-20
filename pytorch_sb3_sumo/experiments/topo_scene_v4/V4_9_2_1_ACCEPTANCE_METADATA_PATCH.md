# v4.9.2.1 acceptance-metadata engineering patch

## Scope

This is an engineering-only patch over the frozen v4.9.2 strict target-only
promotion. It does not change the model, optimizer, reward, collision label,
training budget, checkpoint candidates, deployment decoder, seeds, scenario,
data partition, or promotion thresholds.

The frozen v4.9.2 runner expected a redundant top-level
`formal_test_accessed` key in `paper_evaluation_detailed.json`. The frozen
writer records the actual access evidence instead:

- `evaluation_split` in the detailed evaluation;
- evaluation split and traffic partition in `evaluation_provenance`;
- requested evaluation split in `arguments.json`;
- the formal-unlock receipt path and verified unlock binding in
  `arguments.json`.

Every successful v4.9.2 promotion run recorded `validation` consistently and
recorded no formal-unlock receipt or binding. The missing redundant Boolean
therefore caused false rejection after successful execution.

## Patch behavior

`tools/run_topo_v4_9_2_1_experiments.py`:

1. reuses all frozen v4.9.2 artifact, selector, model-integrity, target-only,
   no-projection, and hash validators;
2. derives formal access from the sealed split/unlock evidence;
3. treats an explicit Boolean, if present, as an additional assertion that
   must agree;
4. reads but never edits the v4.9.2 run tree;
5. writes its summary, manifest, report, and gate only below
   `results_topo_v4_9_2_promotion/engineering_patch_v4_9_2_1/`.

`tools/run_all_pending_promotions_v4_9_2_1.py` is the new versioned registry
entry point. A failed job or version never cancels later jobs or versions; the
registry summarizes only after every registered promotion automation returns.
It never starts formal testing.

## Scientific invariants

- Strict target-only deployment remains mandatory.
- No fusion confidence threshold is admitted.
- No TTC/headway threshold, lane-change veto, geometry-derived unsafe label,
  kinematic action projection, or post-decoder action rewrite is admitted.
- Formal test remains locked unless the complete real promotion matrix passes
  every frozen relative-performance and model-integrity gate.

