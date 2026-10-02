"""Frozen matrix and optimizer settings for the multi-seed confirmation.

Nothing in this module is derived at run time: the three methods, the single
pre-registered candidate each of them carries, the training seeds and the
evaluation seeds are all sealed before any new outcome is observed.
"""
from __future__ import annotations

SCENES = ('left_turn', 'cross', 'roundabout_easy', 'roundabout_medium', 'roundabout', 'carla')
TRAINING_SEEDS = (0, 1, 2)
STEPS = (10000, 20000, 30000, 40000, 50000)

# Optimizer settings are copied verbatim from
# results_phase2_diagnosis_20260908/tuning_contract_v2.json (candidates control,
# lr_half, tau_half).  MST+SLT is the frozen baseline and therefore always runs
# the control candidate.
METHODS = {
    'mst_slt': {'candidate': 'control', 'learning_rate': 1e-4, 'tau': .005,
                'display': 'MST+SLT default'},
    'v4_8': {'candidate': 'lr_half', 'learning_rate': 5e-5, 'tau': .005,
             'display': 'v4.8 lr_half'},
    'v4_13': {'candidate': 'tau_half', 'learning_rate': 1e-4, 'tau': .0025,
              'display': 'v4.13 tau_half'},
}

TRAINING_BUDGET = 50000
TRAINING_WARMUP = 5000
TRAINING_BATCH = 32
TRAINING_BUFFER = 20000
ACTION_REPEAT = 3
CHECKPOINT_FREQUENCY = 10000

SMOKE_BUDGET = 72
SMOKE_WARMUP = 48
SMOKE_BATCH = 2
SMOKE_BUFFER = 128
SMOKE_CHECKPOINT_FREQUENCY = 36

SELECTION = {'steps': list(STEPS), 'episodes_per_checkpoint': 100, 'seed_start': 310000,
             'order': ['success_desc', 'collision_asc', 'timeout_asc', 'later_checkpoint']}
SCORE = {'episodes': 100, 'seed_start': 320000, 'models': ['selected', 'exact_final']}

# Seed-0 cells whose training, unified selection and scoring were already
# completed and sealed by the frozen phase-2 screen run and the phase-3 run.
# They are bound by sha256 instead of being retrained.  `mst_slt` is excluded
# on purpose: its historical seed-0 models come from two different trainer
# lineages, so the aligned trainer has to retrain it.
REUSED_SEED_ZERO_METHODS = ('v4_8', 'v4_13')


def cell_id(method: str, scene: str, seed: int) -> str:
    return f'{method}__{scene}__{METHODS[method]["candidate"]}__seed{seed}'


def phase3_cell_id(method: str, scene: str) -> str:
    return f'{method}__{scene}__{METHODS[method]["candidate"]}'


def is_reused(method: str, seed: int) -> bool:
    return seed == 0 and method in REUSED_SEED_ZERO_METHODS


def cells():
    """The full 54-cell matrix in a deterministic order."""
    return [(method, scene, seed) for method in METHODS for scene in SCENES
            for seed in TRAINING_SEEDS]
