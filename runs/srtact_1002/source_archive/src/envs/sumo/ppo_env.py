"""Source-contract top-down RGB environment for the released SMARTS PPO."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw

from .paper_env import PaperSumoSceneEnv
from .sumo_env import _front_bumper_to_center


_BACKGROUND = (0, 0, 0)
_ROAD = (80, 80, 80)
_EGO = (210, 30, 30)
_SOCIAL = (192, 192, 192)


@dataclass(frozen=True)
class _LaneRasterGeometry:
    points: np.ndarray
    width: float
    bounds: tuple[float, float, float, float]


class _SourceRewardScaler:
    """The released ``ZFilter(shape=(), center=False, gamma=None)``."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.sum_squares = 0.0

    def __call__(self, value: float) -> float:
        value = float(value)
        self.count += 1
        if self.count == 1:
            self.mean = value
        else:
            old_mean = self.mean
            self.mean = old_mean + (value - old_mean) / self.count
            self.sum_squares += (value - old_mean) * (value - self.mean)
        variance = (
            self.sum_squares / (self.count - 1)
            if self.count > 1
            else self.mean * self.mean
        )
        standard_deviation = math.sqrt(max(variance, 0.0))
        difference = value - self.mean
        return float(difference / (standard_deviation + 1e-8) + self.mean)


def _parse_lane_raster_geometry(network_path: Path) -> tuple[_LaneRasterGeometry, ...]:
    root = ET.parse(network_path).getroot()
    lanes: list[_LaneRasterGeometry] = []
    for lane in root.iter("lane"):
        shape_text = lane.attrib.get("shape")
        if not shape_text:
            continue
        points = np.asarray(
            [
                [float(coordinate) for coordinate in point.split(",")[:2]]
                for point in shape_text.split()
            ],
            dtype=np.float64,
        )
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
            continue
        width = float(lane.attrib.get("width", "3.2"))
        lanes.append(
            _LaneRasterGeometry(
                points=points,
                width=width,
                bounds=(
                    float(points[:, 0].min()),
                    float(points[:, 1].min()),
                    float(points[:, 0].max()),
                    float(points[:, 1].max()),
                ),
            )
        )
    if not lanes:
        raise ValueError(f"No lane geometry found in SUMO network {network_path}")
    return tuple(lanes)


class PaperPpoRgbEnv(PaperSumoSceneEnv):
    """80x80 RGB observation used by the released SMARTS PPO branch.

    SMARTS v0.4.17 uses ``RGB(80, 80, 32/80)``: a 32-m square orthographic
    camera centered on and aligned with the ego vehicle.  The source PPO
    runner divides ordinary frames by 255, but accidentally leaves the first
    reset frame of every training episode after episode zero unscaled.  The
    ``training`` flag preserves that behavior while formal evaluation always
    returns normalized frames.
    """

    image_width = 80
    image_height = 80
    resolution = 32.0 / 80.0

    def __init__(self, *args: Any, training: bool = True, **kwargs: Any) -> None:
        kwargs.setdefault("action_repeat", 1)
        super().__init__(*args, **kwargs)
        if self.specification.source_observation_contract != "smarts":
            raise ValueError("PaperPpoRgbEnv is only the released SMARTS PPO contract")
        if self.action_repeat != 1:
            raise ValueError("The released SMARTS PPO chooses an action every raw 0.1-s step")
        self.training = bool(training)
        self.observation_space = spaces.Box(
            low=0.0,
            high=255.0,
            shape=(self.image_height, self.image_width, 3),
            dtype=np.float32,
        )
        self._lane_geometry = _parse_lane_raster_geometry(
            self._paper_specification.network_path
        )
        self._reward_scaler = _SourceRewardScaler()
        self._training_reset_count = 0
        self._last_rgb: np.ndarray | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        _, info = super().reset(seed=seed, options=options)
        self._reward_scaler.reset()
        image = self._render_top_down_rgb()
        normalize = not self.training or self._training_reset_count == 0
        source_reset_scale_bug = bool(self.training and not normalize)
        self._training_reset_count += 1
        observation = image.astype(np.float32)
        if normalize:
            observation /= 255.0
        info = dict(info)
        info["source_top_down_rgb"] = "RGB(80,80,32/80)"
        info["source_reset_scale_bug_applied"] = source_reset_scale_bug
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        _, reward, terminated, truncated, info = super().step(action)
        image = self._render_top_down_rgb().astype(np.float32) / 255.0
        info = dict(info)
        info["source_top_down_rgb"] = "RGB(80,80,32/80)"
        info["source_reset_scale_bug_applied"] = False
        info["source_unscaled_reward"] = float(reward)
        if self.training:
            reward = self._reward_scaler(float(reward))
            # The source runner calls finish_horizon(last_val=0) for every
            # max-step event, despite rewriting its replay done flag. Present
            # it as terminal to SB3 so it does not add a time-limit bootstrap.
            if truncated:
                terminated, truncated = True, False
        return image, float(reward), terminated, truncated, info

    def _world_to_pixel(
        self, points: np.ndarray, center: np.ndarray, sumo_angle_degrees: float
    ) -> np.ndarray:
        angle = math.radians(float(sumo_angle_degrees))
        forward = np.asarray((math.sin(angle), math.cos(angle)), dtype=np.float64)
        right = np.asarray((math.cos(angle), -math.sin(angle)), dtype=np.float64)
        delta = np.asarray(points, dtype=np.float64) - center[None, :]
        local_right = delta @ right
        local_forward = delta @ forward
        return np.column_stack(
            (
                (self.image_width - 1) * 0.5 + local_right / self.resolution,
                (self.image_height - 1) * 0.5 - local_forward / self.resolution,
            )
        )

    def _render_top_down_rgb(self) -> np.ndarray:
        ego_id = self.specification.ego_id
        if (
            self._connection is None
            or ego_id not in self._connection.vehicle.getIDList()
        ):
            if self._last_rgb is not None:
                return self._last_rgb.copy()
            return np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

        ego_front = self._connection.vehicle.getPosition(ego_id)
        ego_angle = float(self._connection.vehicle.getAngle(ego_id))
        ego_length, _ = self._vehicle_dimensions(ego_id)
        center = _front_bumper_to_center(ego_front, ego_angle, ego_length)
        image = Image.new("RGB", (self.image_width, self.image_height), _BACKGROUND)
        draw = ImageDraw.Draw(image)
        crop_radius = 0.5 * 32.0 * math.sqrt(2.0) + 3.0

        for lane in self._lane_geometry:
            min_x, min_y, max_x, max_y = lane.bounds
            if (
                max_x < center[0] - crop_radius
                or min_x > center[0] + crop_radius
                or max_y < center[1] - crop_radius
                or min_y > center[1] + crop_radius
            ):
                continue
            pixels = self._world_to_pixel(lane.points, center, ego_angle)
            draw.line(
                [tuple(point) for point in pixels],
                fill=_ROAD,
                width=max(1, int(round(lane.width / self.resolution))),
                joint="curve",
            )

        for vehicle_id in self._connection.vehicle.getIDList():
            if vehicle_id in self._actors_hidden_this_observation:
                continue
            try:
                front = self._connection.vehicle.getPosition(vehicle_id)
                angle = float(self._connection.vehicle.getAngle(vehicle_id))
                length, width = self._vehicle_dimensions(vehicle_id)
            except self._traci.TraCIException:
                continue
            vehicle_center = _front_bumper_to_center(front, angle, length)
            self._draw_actor_box(
                draw,
                vehicle_center,
                angle,
                length,
                width,
                center,
                ego_angle,
                _EGO if vehicle_id == ego_id else _SOCIAL,
            )

        if self.specification.include_pedestrians:
            for person_id in self._connection.person.getIDList():
                try:
                    position = np.asarray(
                        self._connection.person.getPosition(person_id), dtype=np.float64
                    )
                    angle = float(self._connection.person.getAngle(person_id))
                except self._traci.TraCIException:
                    continue
                self._draw_actor_box(
                    draw,
                    position,
                    angle,
                    0.5,
                    0.5,
                    center,
                    ego_angle,
                    _SOCIAL,
                )

        output = np.asarray(image, dtype=np.uint8).copy()
        self._last_rgb = output
        return output

    def _draw_actor_box(
        self,
        draw: ImageDraw.ImageDraw,
        actor_center: np.ndarray,
        actor_angle: float,
        length: float,
        width: float,
        camera_center: np.ndarray,
        camera_angle: float,
        color: tuple[int, int, int],
    ) -> None:
        actor_radians = math.radians(float(actor_angle))
        forward = np.asarray(
            (math.sin(actor_radians), math.cos(actor_radians)), dtype=np.float64
        )
        right = np.asarray((forward[1], -forward[0]), dtype=np.float64)
        corners = np.asarray(
            [
                actor_center + forward * along + right * across
                for along, across in (
                    (0.5 * length, 0.5 * width),
                    (0.5 * length, -0.5 * width),
                    (-0.5 * length, -0.5 * width),
                    (-0.5 * length, 0.5 * width),
                )
            ]
        )
        pixels = self._world_to_pixel(corners, camera_center, camera_angle)
        draw.polygon([tuple(point) for point in pixels], fill=color)


class PaperPpoCarlaEnv(PaperSumoSceneEnv):
    """Raw-step vector environment used by the released CARLA PPO runner."""

    def __init__(self, *args: Any, training: bool = True, **kwargs: Any) -> None:
        kwargs.setdefault("action_repeat", 1)
        super().__init__(*args, **kwargs)
        if self.specification.source_observation_contract != "carla":
            raise ValueError("PaperPpoCarlaEnv is only the released CARLA PPO contract")
        if self.action_repeat != 1:
            raise ValueError(
                "CARLA PPO stores every raw step while holding each action for three steps"
            )
        self.training = bool(training)
        self.observation_space = spaces.Box(
            low=-1000.0,
            high=1000.0,
            shape=(6, 10, 5),
            dtype=np.float32,
        )
        self._reward_scaler = _SourceRewardScaler()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        self._reward_scaler.reset()
        info = dict(info)
        info["source_ppo_action_hold"] = 3
        return observation["trajectory"], info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        info = dict(info)
        info["source_unscaled_reward"] = float(reward)
        info["source_ppo_action_hold"] = 3
        if self.training:
            reward = self._reward_scaler(float(reward))
        return observation["trajectory"], float(reward), terminated, truncated, info
