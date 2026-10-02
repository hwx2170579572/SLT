"""Source-backed feature extractors for paper baselines."""

from __future__ import annotations

import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .features import keras_initialize_linear
from .ppo import KerasGru


class SourceSacLstmExtractor(BaseFeaturesExtractor):
    """Recover the paper's SAC-"LSTM" state encoder from released code.

    The paper calls this baseline LSTM, while ``RLEncoder`` in the release
    actually instantiates ``tf.keras.layers.GRU(256, return_sequences=True)``
    followed by ``Dense(256, relu)``.  Its intended ``STATE_LSTM`` adapter
    transposes actor histories to ``[time, actors * 5]``.  Ego fields are
    ``[x,y,speed,0,heading]`` and social-actor fields are
    ``[x,y,speed,distance_to_ego,heading]``.  The release omits the runnable
    baseline configuration, so this extractor makes that latent contract
    explicit and deliberately ignores Proposed trajectory/map observations.
    """

    implementation_id = "paper_sac_lstm_reconstruction_v2"

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
    ) -> None:
        if "state_lstm" not in observation_space.spaces or (
            "state_lstm_mask" not in observation_space.spaces
        ):
            raise ValueError(
                "The reconstructed SAC-LSTM requires explicit state_lstm "
                "and state_lstm_mask observations"
            )
        state_space = observation_space.spaces["state_lstm"]
        mask_space = observation_space.spaces["state_lstm_mask"]
        if len(state_space.shape) != 2 or state_space.shape[1] % 5:
            raise ValueError(
                "state_lstm must have shape [time,actors*5], got "
                f"{state_space.shape}"
            )
        history_steps, input_dim = (int(value) for value in state_space.shape)
        if tuple(mask_space.shape) != (history_steps,):
            raise ValueError(
                f"state_lstm_mask must have shape ({history_steps},), got "
                f"{mask_space.shape}"
            )
        if features_dim != 256:
            raise ValueError("The released RLEncoder fixes SAC-LSTM width to 256")
        super().__init__(observation_space, features_dim=features_dim)
        self.actor_count = input_dim // 5
        self.history_steps = history_steps
        self.input_dim = input_dim
        self.gru = KerasGru(input_dim, 256)
        self.output = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.output.apply(keras_initialize_linear)

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        sequence = observations["state_lstm"].float()
        expected = (self.history_steps, self.input_dim)
        if sequence.ndim != 3 or tuple(sequence.shape[1:]) != expected:
            raise ValueError(
                f"Expected [batch,{expected}], got {tuple(sequence.shape)}"
            )
        valid = observations["state_lstm_mask"].bool()
        return self.output(self.gru(sequence, mask=valid))
