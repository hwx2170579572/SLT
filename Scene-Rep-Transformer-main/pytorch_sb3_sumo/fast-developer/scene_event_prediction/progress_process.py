"""A discrete monotone progress process, not independent per-zone ETA heads.

State s < S-1 denotes route progress s * spacing_m. The last state is an
absorbing overflow bucket [cap_m, infinity), not a point mass at cap_m.
Transitions model nonnegative progress increments over a fixed time grid. This
is an explicit forward-driving approximation; lateral/out-of-set alternatives
must be handled by the caller rather than projected silently onto this process.

Only engineering primitives live here. Factual route assignment, censor labels,
shared-stem auxiliary optimization, calibration, and SAC integration remain
separate Stage 5 gates. No M0/M1 execution imports this package.
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ProgressGrid:
    spacing_m: float = 0.5
    states: int = 181
    steps: int = 10
    step_seconds: float = 0.3
    max_advance_bins: int = 18

    def __post_init__(self):
        if (not math.isfinite(self.spacing_m) or not math.isfinite(self.step_seconds)
                or self.spacing_m <= 0 or self.step_seconds <= 0):
            raise ValueError("Grid spacing and time step must be positive")
        if self.states < 2 or self.steps < 1 or self.max_advance_bins < 1:
            raise ValueError("Progress grid has no usable states/transitions")

    @property
    def cap_m(self):
        return (self.states - 1) * self.spacing_m


class ProgressTransitionHead(nn.Module):
    """Decode P(increment | current progress, future time, branch context).

    context shape is [..., width], typically [B, actor, route, width]. The same
    head can later accept a scene-mode axis; it does not introduce scene modes
    or learn route probabilities on its own.
    """

    def __init__(self, width: int, grid: ProgressGrid, hidden: int = 64):
        super().__init__()
        self.grid = grid
        self.context = nn.Linear(width, hidden)
        self.coordinates = nn.Linear(2, hidden, bias=False)
        self.output = nn.Sequential(nn.SiLU(), nn.Linear(hidden, grid.max_advance_bins + 1))
        # Fixed physical scales, rather than normalizing by the chosen grid:
        # changing spacing or dt must change the decoder's physical context.
        t = torch.arange(1, grid.steps + 1, dtype=torch.float32) * grid.step_seconds / 3.0
        s = torch.arange(grid.states, dtype=torch.float32) * grid.spacing_m / 90.0
        tt, ss = torch.meshgrid(t, s, indexing="ij")
        self.register_buffer("grid_coordinates", torch.stack((tt, ss), dim=-1))

    def forward(self, context: Tensor) -> Tensor:
        latent = self.context(context)[..., None, None, :]
        return self.output(latent + self.coordinates(self.grid_coordinates))


class ProgressProcess:
    """Forward filtering over one route, with explicit unobserved integration."""

    def __init__(self, logits: Tensor, grid: ProgressGrid):
        expected = (grid.steps, grid.states, grid.max_advance_bins + 1)
        if tuple(logits.shape[-3:]) != expected:
            raise ValueError(f"Expected final axes {expected}, got {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise ValueError("Transition logits must be finite")
        self.grid = grid
        self.probabilities = logits.softmax(-1)
        source = torch.arange(grid.states, device=logits.device)[:, None]
        advance = torch.arange(grid.max_advance_bins + 1, device=logits.device)[None, :]
        self.destination = (source + advance).clamp_max(grid.states - 1)

    def initial(self) -> Tensor:
        result = self.probabilities.new_zeros(self.probabilities.shape[:-3] + (self.grid.states,))
        result[..., 0] = 1.0
        return result

    def advance(self, distribution: Tensor, time_index: int) -> Tensor:
        if not 0 <= time_index < self.grid.steps:
            raise IndexError(time_index)
        weights = distribution[..., :, None] * self.probabilities[..., time_index, :, :]
        indices = self.destination.expand(weights.shape).flatten(-2)
        return torch.zeros_like(distribution).scatter_add(-1, indices, weights.flatten(-2))

    def marginals(self) -> Tensor:
        """Return [..., T+1, S], including observed current progress zero."""
        current = self.initial()
        states = [current]
        for t in range(self.grid.steps):
            current = self.advance(current, t)
            states.append(current)
        return torch.stack(states, dim=-2)

    def filtered_log_likelihood(self, emission: Tensor, observed: Tensor):
        """Integrate missing times; do not manufacture negative event labels.

        emission [..., T, S] is a supplied nonnegative observation likelihood.
        observed [..., T] marks factual labels, not horizon availability. Missing
        future samples multiply by one. A wholly unobserved sequence has logp=0
        and observed_count=0 and must be skipped by the auxiliary-loss caller.
        Impossible labels retain logp=-inf; they are reported, not smoothed away.
        """
        expected = self.probabilities.shape[:-1]
        if emission.shape != expected or observed.shape != expected[:-1]:
            raise ValueError("Emission and observation axes do not match the process")
        if observed.dtype != torch.bool:
            raise TypeError("Observed mask must be boolean")
        if not torch.isfinite(emission).all() or (emission < 0).any():
            raise ValueError("Observation likelihoods must be finite and nonnegative")
        posterior = self.initial()
        logp = posterior.new_zeros(posterior.shape[:-1])
        impossible = torch.zeros_like(logp, dtype=torch.bool)
        for t in range(self.grid.steps):
            prediction = self.advance(posterior, t)
            weight = torch.where(observed[..., t, None], emission[..., t, :], 1.0)
            unnormalized = prediction * weight
            mass = unnormalized.sum(-1)
            impossible = impossible | (mass <= 0)
            # Do not evaluate log(0) in the differentiated branch. A route can
            # have zero likelihood while another compatible route is valid;
            # its zero mixture weight must not produce 0 * inf gradient NaNs.
            safe_mass = torch.where(mass > 0, mass, 1.0)
            log_mass = torch.where(mass > 0, safe_mass.log(), -torch.inf)
            logp = logp + log_mass
            posterior = unnormalized / safe_mass[..., None]
        return {"log_likelihood": logp, "observed_count": observed.sum(-1),
                "impossible": impossible, "posterior": posterior}

    def two_time_region_probability(self, first_time: int, first_region: Tensor,
                                    second_time: int, second_region: Tensor) -> Tensor:
        """Exact two-time joint under this process, not product of marginals."""
        if not 0 <= first_time <= second_time <= self.grid.steps:
            raise ValueError("Expected 0 <= first_time <= second_time <= horizon")
        expected = self.probabilities.shape[:-3] + (self.grid.states,)
        if first_region.shape != expected or second_region.shape != expected:
            raise ValueError("Region axes must match process state axes")
        if first_region.dtype != torch.bool or second_region.dtype != torch.bool:
            raise TypeError("Region masks must be boolean")
        joint = self.marginals()[..., first_time, :] * first_region
        for t in range(first_time, second_time):
            joint = self.advance(joint, t)
        return (joint * second_region).sum(-1)


class EventDistributionProjector:
    """Project zone thresholds through the same monotone route process.

    Thresholds must already describe *vehicle-center* entry/clearance distances
    for the actor footprint, with shape [..., zones]. No length/width is added a
    second time. A threshold beyond the overflow bucket cannot be resolved and
    receives its own valid mask. Invalid clearance does not erase valid entry.
    First-event masses only describe (now, horizon]; initially entered/cleared
    mass is reported separately, not mislabelled as a new future event at t=0.
    """

    def __call__(self, process: ProgressProcess, entry_m: Tensor, clear_m: Tensor,
                 geometry_valid: Tensor):
        prefix = process.probabilities.shape[:-3]
        if entry_m.shape != clear_m.shape or entry_m.shape[:-1] != prefix:
            raise ValueError("Threshold axes must match process branch axes")
        if geometry_valid.shape != entry_m.shape or geometry_valid.dtype != torch.bool:
            raise ValueError("Geometry mask must match thresholds and be boolean")
        if (geometry_valid & (~torch.isfinite(entry_m) | ~torch.isfinite(clear_m))).any():
            raise ValueError("Valid geometry requires finite thresholds")
        if (geometry_valid & (clear_m < entry_m)).any():
            raise ValueError("Clearance precedes entry")
        marginal = process.marginals()
        positions = torch.arange(process.grid.states, device=entry_m.device,
                                 dtype=entry_m.dtype) * process.grid.spacing_m
        entry_valid = geometry_valid & (entry_m <= process.grid.cap_m)
        clear_valid = geometry_valid & (clear_m <= process.grid.cap_m)
        entry_region = positions >= entry_m[..., None]
        clear_region = positions >= clear_m[..., None]
        entered = (marginal[..., None, :, :] * entry_region[..., :, None, :]).sum(-1)
        cleared = (marginal[..., None, :, :] * clear_region[..., :, None, :]).sum(-1)
        entered = torch.where(entry_valid[..., None], entered, 0.0)
        cleared = torch.where(clear_valid[..., None], cleared, 0.0)
        occupancy_valid = entry_valid & clear_valid
        occupancy = torch.where(occupancy_valid[..., None], (entered - cleared).clamp_min(0), 0.0)
        return {
            "entry_valid": entry_valid, "clear_valid": clear_valid,
            "occupancy_valid": occupancy_valid,
            "entered_cdf": entered, "cleared_cdf": cleared, "occupancy": occupancy,
            "first_entry_after_now": (entered[..., 1:] - entered[..., :-1]).clamp_min(0),
            "first_clear_after_now": (cleared[..., 1:] - cleared[..., :-1]).clamp_min(0),
            "initially_entered": entered[..., 0], "initially_cleared": cleared[..., 0],
            "entry_horizon_survival": torch.where(entry_valid, 1 - entered[..., -1], 0.0),
            "clear_horizon_survival": torch.where(clear_valid, 1 - cleared[..., -1], 0.0),
        }
