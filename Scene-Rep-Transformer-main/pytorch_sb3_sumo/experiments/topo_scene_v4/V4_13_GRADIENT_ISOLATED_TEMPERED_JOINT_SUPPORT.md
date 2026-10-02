# v4.13: Gradient-Isolated Tempered Joint Support PRCR

Status: preregistered before v4.13 implementation or training.

## Problem localized by v4.12

v4.12 improved the two Cross development seeds from the v4.11 references by
`+0.15/-0.15` and `+0.05/-0.05` success/collision, and retained CARLA at
`1.00/0.00`. It failed only `roundabout_medium`, at `0.80/0.20` versus the
v4.11 aggregate reference `0.90/0.10`.

The failure is not assigned to a kinematic safety mechanism: none exists in
the implementation or runtime trace. It is also not fixed by changing the
continuous deployment lane prior. On the same checkpoint and exact same
validation seeds, removing the prior changes D2 by `-0.15/+0.15` and D3 by
`-0.20/+0.20`; increasing it globally from `0.05` to `0.10` leaves D3 at
`0.80/0.20` and fails its locked sensitivity gate.

Real-state autograd localizes the train-time coupling. Both lane categorical
NLL and conditional speed mixture NLL update `actor.latent_pi`; their gradient
norm means are `0.582630` and `0.469633`, respectively. The lane term is thus
`1.240609x` the speed term before outer weighting, and `38.4429%` of jointly
active scalar coordinates point in opposing directions. Both support losses
already have exactly zero gradient in the shared scene encoder. Corroborating
training evidence shows the v4.12 speed-component spread is only `37.98%` of
v4.11 on average and `31.64%` at the final update.

These observations support a narrower hypothesis: replay-lane imitation is
useful for the deployed lane distribution, but its full-strength gradient
through the actor latent trunk perturbs the representation shared by learned
speed components and changes the on-policy data-collection feedback loop.

## Model change

The deployed actor remains one learned hybrid model. For RL and inference it
computes the normal lane logits

`lane_logits = lane_head(latent_pi(scene_features))`.

For the auxiliary replay-lane categorical NLL only, v4.13 recomputes logits
with the same deployed lane-head weights but a detached actor latent:

`support_lane_logits = lane_head(stop_gradient(latent_pi(scene_features)))`.

Consequently:

- replay-lane NLL still calibrates the actual deployed lane head;
- replay-lane NLL has zero gradient in `latent_pi`, component logits, speed
  mean/log-standard-deviation heads, and the scene encoder;
- conditional speed mixture NLL continues to train `latent_pi` and the learned
  speed-mixture heads;
- no new inference head, rule, threshold, or action transformation is added.

The lane auxiliary scale inside the joint support loss is globally tempered
from `1.0` to `0.25`:

`joint_support_nll = conditional_speed_mixture_nll + 0.25 * lane_nll`.

The outer replay-support coefficient remains `0.05`. The inference lane prior
remains `0.05`; the component prior remains `0.05`. The exact learned
reward-risk-uncertainty score and feasible physical lane-existence mask are
unchanged from v4.12.

## What remains unchanged

- observation, action space, reward, and source traffic protocol;
- topology-temporal Graph-SLT scene encoder;
- Cross-only extension of the existing random-rotation training augmentation;
- learned collision critic, proper soft Bernoulli loss, pairwise rank
  coefficient `0.25`, temporal consistency coefficient `0.05`;
- collision-risk coefficient `1.0`, twin-uncertainty coefficient `0.25`;
- three learned conditional speed components per lane and no fixed speed grid;
- observed environment collision labels, no future/oracle labels, n-step `16`;
- train-only checkpoint selection and untouched formal-test split.

## Explicitly forbidden

No TTC/headway threshold, geometry-derived unsafe label, lane-change veto,
confidence gate, rule fallback, safety shield, semantic tie override,
scenario-conditioned inference coefficient, post-decoder action rewrite, or
external/internal kinematic safety projection may be introduced. Physical
adjacent-lane existence masking remains a structural action-space constraint,
not a traffic-safety rule.

## Evidence plan

Engineering must prove exact gradient ownership, model save/load fidelity,
exact learned argmax/action equality, no-rule/no-projection static and dynamic
audits, and a real SUMO smoke run.

Fresh development uses the exact v4.12 D1--D4 train/evaluation seed contracts
so deltas are directly paired. Every primary cell is attempted before the gate
is summarized. The method must retain both Cross cells within `0.05`, improve
D3 by at least one 20-episode outcome step to at least `0.85/0.15`, and retain
CARLA within `0.05`. Only a complete pass unlocks the three preregistered
ablations, then fresh 12-job promotion. Formal test remains locked until the
complete promotion gate passes.

No outcome improvement is claimed by this design document; all missing or
failed evidence remains TBD.
