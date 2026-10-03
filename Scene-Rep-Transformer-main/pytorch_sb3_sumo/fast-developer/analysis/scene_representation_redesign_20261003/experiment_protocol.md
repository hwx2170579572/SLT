# Scene-event representation: independent implementation and experiment protocol

Status: implementation in progress; no new performance result is asserted by this document.
Date: 2026-10-03. Design source: [implementation roadmap](implementation_roadmap.md).

## Scope and confirmed task semantics

The user authorized implementation of the complete staged roadmap, stage-by-stage acceptance checks, and repair of failed implementation checks. The main scene is `intersection_sorted_depart4p0`. Existing methods, checkpoints, numeric results, and legacy training protocols remain identifiable historical evidence. New methods use an independent entry point and protocol version; an implementation check is not evidence of a success-rate improvement.

**User-confirmed 2026-10-03:** 60 seconds is a task deadline. New observations include remaining task time; expiration is a true terminal transition, so no value bootstrap is taken after it. External collection interruption is a separate truncation and is never silently relabelled as task failure. Warm-up is not part of the ego-control deadline.

## Common SAC protocol

- Protocol ID: `scene_event_sac_deadline_nstep_v1`.
- Physics time unit: 0.1 s. Decision action repeat: 3 physics steps, shortened at termination.
- Discount is defined per decision. An n-step transition contains the discounted rewards for its actual `k` decisions and bootstraps with `gamma ** k`, with `k <= 4`.
- To keep the remaining learning mechanism fixed, the target retains the historical **reward-only multi-step sum plus endpoint soft bootstrap** convention. It does not accumulate intermediate entropy terms and does not apply importance sampling/Retrace corrections for replay behavior. This is an approximate off-policy target, not an exact multi-step soft-policy-evaluation operator. Correcting `gamma ** k` and deadline semantics does not remove this separate approximation. Any later comparison to one-step SAC or a corrected multi-step algorithm must be registered as a common learning-protocol experiment.
- Six environment reward components and their environment/raw aggregates are retained and reconciled. Representation changes do not alter the reward function.
- Shared defaults for the new M0/M1 arms: learning rate 1e-4, batch 32, replay capacity 10000 decision transitions, NAdam with eps 1e-7, initial entropy coefficient 0.2, gamma 0.99, tau 0.005, policy/Q hidden widths 128/32, 64-dimensional action embedding, and learning starts at 5000 raw steps. Replay remains uniform-sampled with FIFO overwrite. The capacity is intentionally smaller than the old local SAC profile for a measured host-memory constraint; it is identical across M0/M1 and is not a performance-tuned value. Exact update scheduling and implementation identity are recorded; new-versus-legacy comparison is not a bitwise replication claim.
- Resource basis: the current Z=90 observation schema has 21 dynamic arrays totaling 221140 bytes per observation. With a 20000-transition replay and one-decision terminal episodes, immutable observation/next-observation snapshots may not overlap across episodes; the static two-worker array bound is 17,691,200,000 bytes, above the 15.15 GB then free on the host before Python/container/model/environment overhead. At 10000 transitions, that same two-worker worst-case array bound is 8,845,600,000 bytes; overhead remains additional and actual memory depends on episode lengths. Longer episodes share immutable observations across overlapping n-step transitions and use less array storage. The failed `deadline01` snapshot retains its original 20000 capacity; the repaired `deadline02` check will use 10000. No historical method, checkpoint, or result is changed.
- After warm-up and availability of one replay batch, execute three updates per replay-eligible decision, including a decision shortened by true termination. Record actual updates rather than inferring them from the raw budget. Each Q has its own action projection.
- The global raw-budget cutoff is external collection truncation, distinct from the per-episode deadline. If that cutoff interrupts a nonterminal three-raw macro-action after only one/two ticks, retain its physical observations/reward in collection logs but omit that incomplete macro-fragment from replay, following the old held-action convention. Flush pending returns at the last **completed decision's actual next observation**, never a reset state or a fabricated combined transition. Record `replay_used_raw` and `omitted_tail_raw` separately. `gamma ** k` counts decisions and does not itself compensate for a partial macro-action's physical duration. A short action ending in true task termination remains eligible and has no bootstrap.
- The entropy-coefficient optimizer follows the local baseline's Adam (`betas=(0.5, 0.999)`, eps 1e-7); NAdam applies to actor and critic/encoder parameters, not the entropy coefficient.
- Common actor and critic heads use separate deterministic initialization streams (`seed+101` and `seed+202`), restoring the ambient RNG afterward. This prevents encoder parameter-count differences alone from shifting the initial common heads. It does not turn one training seed into evidence of robustness.
- The inherited environment already discounts within a repeated action with `reward_discount=0.99`; replay then discounts across decisions. This inherited two-level reward convention is retained and disclosed. The new `gamma ** k` exponent counts decisions, not physical ticks. It is not represented as a single continuous-time discount model.
- Both critics receive `(z_t, a_t)`. The actor receives `z_t`; its feature input is detached in the initial protocol.
- Online observations and replay observations pass through the current encoder. Target action uses the online encoder/current actor; target critics use a separately synchronized target encoder. No cached historical embedding is used as a replay state.
- A shared encoder belongs to one optimizer and is updated once by the combined TD plus eligible auxiliary loss. Prediction heads receive auxiliary gradients only, event token/graph readout receives TD gradients, and target parameters receive no gradients.
- Learned progress/event/mode distributions are detached on their path into the SAC event branch. Shared-stem TD gradients still indirectly affect predictions; calibration is therefore measured, not assumed.
- Initial encoder dropout is zero; SAC still samples stochastic actions for training. Frozen consistency checks compare distribution parameters or deterministic means.

## Information contract

Current/past observable vehicle kinematics and public static lane connectivity are allowed. Ego navigation is allowed. Social future routes, future departures, full future trajectories, and simulator IDs as learned features are excluded. Tracking keys can join histories/labels but are not network inputs. Future labels come from later normal observations only. An actor disappearing or falling outside coverage is censored, not interpreted as a free conflict region.

Initial settings from the roadmap are engineering defaults, not tuned findings: radius 80 m, maximum 24 actors including ego, 21 history samples over 2 s, width and output dimension 128, prediction horizon 3 s at 0.3 s resolution. Any change must be recorded with its reason and used equally in the relevant comparison arms.

## Staged acceptance ledger

Each stage has separate implementation and empirical status. Only recorded passing checks may change `pending` to `passed`. A failed meaningful test must be fixed, not skipped or weakened. Failure of a research hypothesis is reported and used to revise the method; a target metric is not guaranteed.

| Stage | Deliverable | Mandatory implementation acceptance | Empirical acceptance | Current status |
|---|---|---|---|---|
| 0 | Frozen common protocol | Actual-k discounted targets, terminal/truncation distinctions, remaining-time contract, optimizer/target ownership tests | Real-environment confirmation deferred to Stage 3 smoke | numeric/ownership unit subgate passed; live integration pending (see stage_acceptance.md) |
| 1 | Observation/history/replay contract | Explicit masks; left/right/internal missing histories; age; no ID/future-route leakage; mature/censored labels; raw-step collection | Real pre-action observation and coverage audit | pending |
| 2 | Public movement/conflict geometry | Successor/internal/exit paths; crossing/merge/following/parallel/static/curved/boundary cases; route/zone order | Real map coverage and construction cost | pending |
| 3 | M0 generic dual graph and SAC | Permutation/padding/target synchronization/replay re-encoding/finite losses/real updates; independent entry point | Bounded real SUMO smoke before full development experiment | pending |
| 4 | M1 deterministic passage events | Consistent entry/clearance/occupancy, route exclusivity, invalid/unknown/stopped states, no future-route oracle | Event availability/use; M0 comparison with information and cost disclosed | pending |
| 5 | M2 learned progress and labels | One progress process per path, factual/censored likelihood, mature-target masks, detached predicted features, no double optimizer update | Calibration, held-out episode data, same predictor/control comparisons | pending |
| 6 | M3 shared scene modes | Same-decoder K1 control; K2/K4; scene likelihood; branch-preserving relational marginalization; permutation and joint-distribution counterexamples | Joint information evidence, calibration, collapse/cost checks | pending |
| 7 | Evidence pipeline | Config/source/checkpoint hashes, exact episode identity, outcome/reward accounting, paired result summaries and claim ledger | Development results and eventually multi-seed/generalization evidence | pending |

## Method identities and comparisons

- M0: `sac_scene_dualgraph_v1`.
- M1: `sac_scene_eventgraph_cv_v1`.
- M2: `sac_scene_eventgraph_pred_v1`.
- M3: `sac_scene_eventgraph_joint_v1`, with explicit K1/K2/K4 configuration.

These are proposed implementation identities until code registration and verification are recorded. New-input models cannot be interpreted as pure architecture ablations against historical five-neighbor methods. M0 to M1 changes event semantics and graph structure; a same-event-features generic-message control is required for the narrower structure claim. M2 versus M3 also needs a same-decoder K1 comparison. Parameter count, throughput, latency, observation coverage and auxiliary data budgets accompany each result.

Development experiments, when the relevant implementation gates pass, use seed 0, fresh training, 100000 ego-control raw steps and final 100 validation episodes with seeds 10000–10099. These familiar seeds are a development set, not an untouched test set. Short correctness smokes live in separate roots and never contribute to final performance tables. Full 3–5 training-seed and generalization studies are later scientific evidence, not something that passing unit tests can establish.

## Normal-process diagnostics

Collect input masks/short histories/actor cutoffs; public path and conflict coverage; explicit unknown/censor reasons; event timing/probability calibration with independent passage IDs; module activations/gradients/actual parameter changes; TD/auxiliary gradient scales; target synchronization; action proposals and executed speed/lane; entry/clearance/waiting and first collision data. Normal shadow probes have no extra simulator steps, at most 20 unique training states in the entire phase at 5000-raw thresholds and at most four unique states per evaluation episode. Variant rows are not episodes. Same-state functional sensitivity is not a performance causal effect.

## Evidence retention

Stage-specific validation artifacts contain the exact command, interpreter, date, exit status, test counts, source hashes, limitations and unresolved checks. Real smoke runs record raw/decision counts and simulator processes, provenance, reward accounting and checkpoint reload identity. Summary tables retain unsuccessful variants and distinguish implementation acceptance from empirical support. Research context/history are updated when this route materially changes.
