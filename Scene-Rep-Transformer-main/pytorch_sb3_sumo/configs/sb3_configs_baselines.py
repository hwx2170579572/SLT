"""Unified registry for the six comparison baselines.

Re-exports every baseline constructor and keeps the ``algo -> model`` mapping
in one auditable place.  The existing frozen ``sb3_configs.py`` branches are
left untouched; ``mst`` (MST without SLT) is wired through its original branch.

Method classification of the six baselines:

=============  ==============  ========  =====================
category        baseline        head      encoder
=============  ==============  ========  =====================
hybrid action   HSAC            hybrid    MLP/LSTM
hybrid action   HyAR            hybrid    MLP (latent VAE)
graph RL        GNN+SAC         continuous vehicle GNN (DGN)
transformer RL  Hybrid-DT       hybrid    GPT-2
transformer RL  Decision Trans. seq       GPT-2
transformer RL  MST (no SLT)    continuous MST transformer
=============  ==============  ========  =====================
"""

from __future__ import annotations

from typing import Any

from algos.graph_rl.gnn_sac import make_gnn_sac_model
from algos.hybrid_action.hsac import make_hsac_model
from algos.hybrid_action.hyar import make_hyar_model
from algos.transformer_rl import make_decision_transformer, make_hybrid_dt
from configs.sb3_configs import make_model

# SB3-native baselines (constructed directly from an environment).
SB3_BASELINE_ALGORITHMS = ("hsac_mlp", "hsac_lstm", "gnn_sac", "mst")

# Sequence / latent baselines (constructed from a flat state dimension).
SEQUENCE_BASELINE_ALGORITHMS = ("hybrid_dt", "decision_transformer", "hyar")

BASELINE_ALGORITHMS = SB3_BASELINE_ALGORITHMS + SEQUENCE_BASELINE_ALGORITHMS


def make_baseline_model(
    algo: str,
    env: Any,
    *,
    scenario: str,
    **kwargs: Any,
) -> Any:
    """Construct an SB3-native baseline model from an environment.

    ``hybrid_dt`` / ``decision_transformer`` / ``hyar`` are not SB3 models and
    must be built through ``make_sequence_baseline`` (which needs a flattened
    state dimension instead of an environment).
    """
    if algo in ("hsac_mlp", "hsac_lstm"):
        return make_hsac_model(algo, env, scenario=scenario, **kwargs)
    if algo == "gnn_sac":
        return make_gnn_sac_model(algo, env, scenario=scenario, **kwargs)
    if algo == "mst":
        return make_model("mst", env, scenario=scenario, **kwargs)
    raise ValueError(
        f"{algo!r} is not an SB3-native baseline; use make_sequence_baseline "
        f"with a flattened state dimension. Choices: {SB3_BASELINE_ALGORITHMS}"
    )


def make_sequence_baseline(
    algo: str,
    state_dim: int,
    *,
    act_dim: int | None = None,
    **kwargs: Any,
) -> Any:
    """Construct a sequence / latent baseline from a flat state dimension."""
    if algo == "hybrid_dt":
        return make_hybrid_dt(state_dim, **kwargs)
    if algo == "decision_transformer":
        if act_dim is None:
            raise ValueError("decision_transformer requires act_dim")
        return make_decision_transformer(state_dim, act_dim, **kwargs)
    if algo == "hyar":
        return make_hyar_model(state_dim, **kwargs)
    raise ValueError(
        f"Unknown sequence baseline {algo!r}. "
        f"Choices: {SEQUENCE_BASELINE_ALGORITHMS}"
    )


BASELINES = {
    "hsac_mlp": {
        "category": "hybrid_action",
        "action_head": "hybrid",
        "encoder": "mlp",
        "reference": "Hybrid Soft Actor-Critic",
    },
    "hsac_lstm": {
        "category": "hybrid_action",
        "action_head": "hybrid",
        "encoder": "lstm",
        "reference": "Hybrid Soft Actor-Critic",
    },
    "hyar": {
        "category": "hybrid_action",
        "action_head": "hybrid",
        "encoder": "latent_vae",
        "reference": "HyAR (ICLR 2022)",
    },
    "gnn_sac": {
        "category": "graph_rl",
        "action_head": "continuous",
        "encoder": "vehicle_gnn",
        "reference": "DGN (ICLR 2020)",
    },
    "hybrid_dt": {
        "category": "transformer_rl",
        "action_head": "hybrid",
        "encoder": "gpt2",
        "reference": "Hybrid Decision Transformer",
    },
    "decision_transformer": {
        "category": "transformer_rl",
        "action_head": "sequence",
        "encoder": "gpt2",
        "reference": "Decision Transformer (NeurIPS 2021)",
    },
    "mst": {
        "category": "transformer_rl",
        "action_head": "continuous",
        "encoder": "mst",
        "reference": "MST without SLT (already implemented)",
    },
}


__all__ = [
    "BASELINES",
    "BASELINE_ALGORITHMS",
    "SB3_BASELINE_ALGORITHMS",
    "SEQUENCE_BASELINE_ALGORITHMS",
    "make_baseline_model",
    "make_sequence_baseline",
]
