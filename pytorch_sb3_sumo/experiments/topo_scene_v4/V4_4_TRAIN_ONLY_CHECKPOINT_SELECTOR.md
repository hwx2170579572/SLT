# v4.4: Train-Only Deployment-Aware Checkpoint Selection

Status: preregistered before implementation and fresh v4.4 development runs.

## Why v4.3 stopped

v4.3 passed R1, R2, T1, and T2, then failed T3 (Cross seed 1): the exact-final
checkpoint produced 0/12 validation successes, 2/12 collisions, and 10/12
timeouts. Decision traces show that all ten timeout episodes had already reached
the terminal edge, so the failure was not primarily an unexecuted lane change.
Their final-edge speed was too low to cover the remaining route before the
source episode limit.

The training-best checkpoint from the same run, frozen at raw step 15,715,
achieved 11/12 successes, 1/12 collision, and no timeout on the identical
validation seeds. Its mean speed was 4.91 m/s versus 4.19 m/s for the final
checkpoint. This isolates late-training longitudinal drift.

Blindly selecting the training-best checkpoint is not safe: on T1 (CARLA seed
2), the training-best checkpoint failed 12/12 while the final checkpoint passed
12/12. The deployable question is therefore checkpoint selection, not a global
preference for an earlier checkpoint.

## Single change

v4.4 retains the complete v4.3 algorithm and training process. It changes only
the checkpoint used for deterministic deployment/evaluation:

1. Candidate checkpoints are the exact-final checkpoint and the callback's
   highest-training-success checkpoint. If the latter was never produced, the
   candidate set contains exact-final only.
2. Both candidates are evaluated on the same 12 episodes from the **training
   traffic partition** using the frozen deterministic deployment decoder.
3. Selection is lexicographic: maximize success count; minimize collision
   count; minimize off-route count; minimize timeout count; maximize mean
   return; prefer exact-final on a complete tie.
4. A selector receipt is sealed before any validation or test environment is
   constructed. Validation and formal-test outcomes can never influence the
   selected checkpoint.

The target-critic lane decoder, stochastic training policy, reward, encoder,
critic learning, traffic split, action repeat, and all optimizer settings are
unchanged from v4.3.

## Post-hoc feasibility evidence (not confirmatory)

Twelve paired train-partition calibration episodes per checkpoint give:

| Historical cell | Exact final | Training best | Selected |
|---|---:|---:|---|
| T1 CARLA seed 2 | 12 success / 0 failure | 0 success / 12 timeout | final |
| T2 Cross seed 0 | 9 success / 3 collision | 9 success / 3 collision | final (tie) |
| T3 Cross seed 1 | 0 success / 2 collision / 10 timeout | 10 success / 2 collision | best |

The resulting choices all pass their independent historical validation gates.
These exposed outcomes justify the mechanism and rule but do not count as v4.4
development confirmation.

## Fresh development sequence

Run sequentially and stop at the first hard-gate failure:

1. D1: CARLA seed 3, 20k raw steps; train calibration 64000–64011;
   validation 55000–55011.
2. D2: Cross seed 2, 20k raw steps; train calibration 65000–65011;
   validation 56000–56011.
3. D3: Cross seed 3, 20k raw steps; train calibration 66000–66011;
   validation 57000–57011.
4. D4: Roundabout-medium seed 0, 20k raw steps; train calibration
   67000–67011; validation 58000–58011.

CARLA retains the v4.3 outcome and target-decoder gates. Cross and
roundabout-medium require success at least 0.50, collision at most 0.45,
off-route 0, and timeout at most 0.30. Every cell must also prove that selection
used exactly two paired train-only calibration blocks (or a documented
single-final fallback), that the receipt predates validation, and that the
evaluated model hash equals the selected checkpoint hash.

## Promotion and formal test

After all four development cells pass, run the same 12-job paired promotion as
v4.3. The checkpoint selector is applied symmetrically to TemporalGraph and the
candidate. Promotion thresholds remain unchanged.

Only a passing promotion receipt unlocks the untouched 120-job formal test.
Both methods use the same train-only selector; formal test seeds and outcomes
remain inaccessible until selection is sealed.

## Infrastructure interruptions

An interruption before a complete accepted run is not a scientific failure.
Archive the immutable partial attempt and permit one identical from-scratch
retry. Partial checkpoints are never eligible. A second matching infrastructure
failure stops the protocol for a versioned engineering attribution.

