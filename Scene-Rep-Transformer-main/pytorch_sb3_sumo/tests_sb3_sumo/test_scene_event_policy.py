from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


FAST_DEVELOPER = Path(__file__).resolve().parents[1] / "fast-developer"
if str(FAST_DEVELOPER) not in sys.path:
    sys.path.insert(0, str(FAST_DEVELOPER))

from scene_event.policy import SceneSAC  # noqa: E402
from scene_event.protocol import ExperimentConfig  # noqa: E402


class TinyEncoder(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.projection = nn.Linear(1, z_dim)
        self.register_buffer("offset", torch.zeros(z_dim))
        self.forward_records: list[dict[str, torch.Tensor | bool]] = []

    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        state = observation["state"].float().reshape(-1, 1)
        z = self.projection(state) + self.offset
        self.forward_records.append(
            {
                "grad_enabled": torch.is_grad_enabled(),
                "state": state.detach().clone(),
                "z": z.detach().clone(),
            }
        )
        return z


def _config(*, tau: float = 0.25) -> ExperimentConfig:
    return ExperimentConfig(
        device="cpu",
        z_dim=4,
        width=4,
        hidden_dim=8,
        final_hidden_dim=4,
        action_embedding_dim=3,
        batch_size=2,
        replay_capacity=8,
        tau=tau,
    )


def _batch() -> dict[str, object]:
    return {
        "observation": {
            "state": np.asarray([[0.2], [-0.4]], dtype=np.float32),
            "remaining_time_s": np.asarray([[20.0], [18.0]], dtype=np.float32),
        },
        "next_observation": {
            "state": np.asarray([[0.5], [0.8]], dtype=np.float32),
            "remaining_time_s": np.asarray([[19.7], [17.7]], dtype=np.float32),
        },
        "action": np.asarray([[0.1, -0.2], [-0.3, 0.4]], dtype=np.float32),
        "reward": np.asarray([[1.0], [-0.5]], dtype=np.float32),
        "discount": np.asarray([[0.99**2], [0.99]], dtype=np.float32),
        "terminated": np.asarray([[False], [True]], dtype=bool),
    }


def test_scene_sac_owns_each_trainable_parameter_once() -> None:
    model = SceneSAC(TinyEncoder(4), _config())
    model.assert_parameter_ownership()
    optimizers = (model.critic_optimizer, model.actor_optimizer, model.alpha_optimizer)
    owned = [
        [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
        for optimizer in optimizers
    ]
    assert len(set(owned[0] + owned[1] + owned[2])) == sum(map(len, owned))
    assert set(owned[0]) == {
        id(parameter)
        for parameter in model.encoder.parameters()
    } | {id(parameter) for parameter in model.critic.parameters()}
    assert set(owned[1]) == {id(parameter) for parameter in model.actor.parameters()}
    assert set(owned[2]) == {id(model.log_alpha)}


def test_update_uses_online_actor_target_encoder_and_detached_actor_features() -> None:
    model = SceneSAC(TinyEncoder(4), _config())
    actor_inputs: list[torch.Tensor] = []
    original_sample = model.actor.sample

    def record_actor_input(z: torch.Tensor, deterministic: bool = False):
        actor_inputs.append(z.detach().clone())
        return original_sample(z, deterministic=deterministic)

    model.actor.sample = record_actor_input  # type: ignore[method-assign]
    target_q_inputs: list[torch.Tensor] = []
    original_target_q = model.target_critic.forward

    def record_target_q(z: torch.Tensor, action: torch.Tensor):
        target_q_inputs.append(z.detach().clone())
        return original_target_q(z, action)

    model.target_critic.forward = record_target_q  # type: ignore[method-assign]

    # Keep TD updates, then clear the encoder gradients at that boundary. Any
    # gradient left at the end would therefore have come from the actor loss.
    original_critic_step = model.critic_optimizer.step

    def critic_step_then_clear_encoder_grad():
        result = original_critic_step()
        for parameter in model.encoder.parameters():
            parameter.grad = None
        return result

    model.critic_optimizer.step = critic_step_then_clear_encoder_grad  # type: ignore[method-assign]

    encoder_before = [parameter.detach().clone() for parameter in model.encoder.parameters()]
    actor_before = [parameter.detach().clone() for parameter in model.actor.parameters()]
    metrics = model.update(_batch(), audit=True)

    assert len(model.encoder.forward_records) == 3
    online_records = model.encoder.forward_records
    assert [record["grad_enabled"] for record in online_records] == [False, True, False]
    assert torch.equal(online_records[0]["state"], torch.tensor([[0.5], [0.8]]))
    assert torch.equal(online_records[1]["state"], torch.tensor([[0.2], [-0.4]]))
    assert torch.equal(online_records[2]["state"], torch.tensor([[0.2], [-0.4]]))

    assert len(model.target_encoder.forward_records) == 1
    target_record = model.target_encoder.forward_records[0]
    assert target_record["grad_enabled"] is False
    assert torch.equal(target_record["state"], torch.tensor([[0.5], [0.8]]))
    assert torch.equal(target_q_inputs[0], target_record["z"])

    assert torch.equal(actor_inputs[0], online_records[0]["z"])
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.target_critic.parameters())
    assert metrics["target_has_gradient"] == 0.0
    assert metrics["encoder_parameter_delta_norm"] > 0.0
    assert all(torch.isfinite(torch.tensor(metrics[key])) for key in ("td_loss", "actor_loss", "alpha_loss"))
    assert any(not torch.equal(before, after) for before, after in zip(actor_before, model.actor.parameters()))


def test_soft_update_covers_encoder_critic_parameters_and_buffers() -> None:
    tau = 0.25
    model = SceneSAC(TinyEncoder(4), _config(tau=tau))
    with torch.no_grad():
        for parameter in model.encoder.parameters():
            parameter.fill_(2.0)
        for parameter in model.critic.parameters():
            parameter.fill_(4.0)
        for parameter in model.target_encoder.parameters():
            parameter.fill_(0.0)
        for parameter in model.target_critic.parameters():
            parameter.fill_(0.0)
        model.encoder.offset.fill_(3.0)
        model.target_encoder.offset.fill_(-1.0)

    model.soft_update()
    for parameter in model.target_encoder.parameters():
        assert torch.allclose(parameter, torch.full_like(parameter, 0.5))
    for parameter in model.target_critic.parameters():
        assert torch.allclose(parameter, torch.full_like(parameter, 1.0))
    assert torch.equal(model.target_encoder.offset, model.encoder.offset)
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.target_critic.parameters())


def test_common_sac_heads_have_equal_initialization_across_encoder_variants() -> None:
    class EncoderWithExtraParameters(TinyEncoder):
        def __init__(self, z_dim: int):
            super().__init__(z_dim)
            self.variant_specific = nn.Linear(3, 5)

    config_m0 = _config()
    config_m1 = ExperimentConfig(**{**config_m0.to_dict(), "method": "sac_scene_eventgraph_cv_v1"})

    torch.manual_seed(31415)
    encoder_m0 = TinyEncoder(config_m0.z_dim)
    rng_before_policy = torch.get_rng_state().clone()
    model_m0 = SceneSAC(encoder_m0, config_m0)
    assert torch.equal(torch.get_rng_state(), rng_before_policy)

    # This encoder consumes a different number of initialization draws. The
    # common actor/Q heads must still have the same seed-controlled weights.
    torch.manual_seed(27182)
    encoder_m1 = EncoderWithExtraParameters(config_m1.z_dim)
    rng_before_policy = torch.get_rng_state().clone()
    model_m1 = SceneSAC(encoder_m1, config_m1)
    assert torch.equal(torch.get_rng_state(), rng_before_policy)

    for left, right in zip(model_m0.actor.state_dict().values(), model_m1.actor.state_dict().values()):
        assert torch.equal(left, right)
    for left, right in zip(model_m0.critic.state_dict().values(), model_m1.critic.state_dict().values()):
        assert torch.equal(left, right)
    assert sum(p.numel() for p in model_m1.encoder.parameters()) > sum(
        p.numel() for p in model_m0.encoder.parameters()
    )
