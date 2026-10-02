"""Read-only raw-step telemetry; missing safety measurements remain missing."""
from __future__ import annotations

import math
import types
from pathlib import Path

import gymnasium as gym
import numpy as np

from .common import sha
from .curves import distribution


def finite(value):
    value = float(value)
    if not math.isfinite(value) or value < -1e6:
        raise ValueError('Invalid TraCI numeric measurement')
    return value


class DrivingTrace(gym.Wrapper):
    """Hook observation after SUMO steps; never modify actions or simulation."""
    def __init__(self, env):
        super().__init__(env)
        self.active = False
        self.samples, self.actions = [], []
        self.attempted_raw_steps = 0
        original = env._after_simulation_step
        command = env._sumo_command

        def after_step(owner):
            original()
            if self.active:
                self.attempted_raw_steps += 1
                self.samples.append(self.sample())

        def capture_command(owner, seed):
            value = command(seed)
            # Capture the actual call. Calling _sumo_command again would advance
            # the traffic-template cycle and invalidate pairing.
            self.command = list(value)
            return value

        env._after_simulation_step = types.MethodType(after_step, env)
        env._sumo_command = types.MethodType(capture_command, env)

    def sample(self):
        connection, ego = self.env._connection, self.env.specification.ego_id
        if ego not in connection.vehicle.getIDList():
            return None
        vehicle = connection.vehicle
        value = dict(time=finite(connection.simulation.getTime()), speed=finite(vehicle.getSpeed(ego)),
                     acceleration=finite(vehicle.getAcceleration(ego)), angle=finite(vehicle.getAngle(ego)),
                     distance=finite(vehicle.getDistance(ego)), allowed_speed=finite(vehicle.getAllowedSpeed(ego)),
                     lateral_speed=finite(vehicle.getLateralSpeed(ego)), road=vehicle.getRoadID(ego),
                     lane=int(vehicle.getLaneIndex(ego)), leader_gap=None, closing_speed=None, ttc=None, headway=None)
        leader = vehicle.getLeader(ego, 100.)
        if leader and leader[0]:
            # SUMO's getLeader gap excludes ego minGap. Restore physical
            # bumper-to-bumper distance, clamped at zero for overlap.
            gap = max(0., finite(leader[1]) + finite(vehicle.getMinGap(ego)))
            closing = value['speed'] - finite(vehicle.getSpeed(leader[0]))
            value.update(leader_gap=gap, closing_speed=closing,
                         ttc=gap / closing if closing > 1e-6 else None,
                         headway=gap / value['speed'] if value['speed'] > .1 else None)
        return value

    def reset(self, **kwargs):
        self.active = False
        self.samples, self.actions = [], []
        self.attempted_raw_steps = 0
        obs, info = self.env.reset(**kwargs)
        self.initial = self.sample()
        self.active = True
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        info = result[-1]
        mask, command = info.get('pre_action_lane_action_mask'), info.get('lane_command')
        feasible = bool(mask[int(command) + 1]) if mask is not None and command is not None else None
        self.actions.append(dict(lane_command=command, feasible=feasible,
                                 target_speed=info.get('effective_target_speed')))
        return result

    def traffic_identity(self):
        command = self.command
        routes = command[command.index('--route-files') + 1].split(',')
        network = command[command.index('--net-file') + 1]
        return dict(seed=int(command[command.index('--seed') + 1]),
                    route_sha256=[sha(Path(p)) for p in routes], network_sha256=sha(Path(network)),
                    route_basenames=[Path(p).name for p in routes], command=command)

    def metrics(self, raw_steps):
        if self.attempted_raw_steps != raw_steps:
            raise ValueError('Telemetry raw-step count mismatch')
        return telemetry_metrics(self.initial, self.samples, self.actions, raw_steps)


def telemetry_metrics(initial, samples, actions, raw_steps):
    observed = [s for s in samples if s is not None]
    result = dict(raw_steps=raw_steps, observed_raw_steps=len(observed),
                  raw_step_coverage=len(observed) / raw_steps if raw_steps else None)
    if not observed:
        raise ValueError('No ego telemetry; evaluation cannot supply driving metrics')
    speed = np.asarray([s['speed'] for s in observed])
    accel = np.asarray([s['acceleration'] for s in observed])
    jerk, lateral, distance = [], [], 0.
    lane_changes = 0
    previous = initial
    for current in samples:
        if current is not None and previous is not None:
            dt = current['time'] - previous['time']
            if dt <= 0:
                raise ValueError('Non-increasing telemetry timestamp')
            jerk.append((current['acceleration'] - previous['acceleration']) / dt)
            angle_delta = (math.radians(current['angle'] - previous['angle']) + math.pi) % (2 * math.pi) - math.pi
            lateral.append(current['speed'] * angle_delta / dt)
            distance += max(0., current['distance'] - previous['distance'])
            lane_changes += int(current['road'] == previous['road'] and current['lane'] != previous['lane'])
        previous = current  # Never bridge missing samples for derivatives/distance.

    def rms(x):
        return float(np.sqrt(np.mean(np.square(x)))) if len(x) else None

    def abs95(x):
        return float(np.percentile(np.abs(x), 95)) if len(x) else None

    ttc = [s['ttc'] for s in observed if s['ttc'] is not None]
    headway = [s['headway'] for s in observed if s['headway'] is not None]
    leader_count = sum(s['leader_gap'] is not None for s in observed)
    feas = [a['feasible'] for a in actions if a['feasible'] is not None]
    result.update(mean_speed_mps=float(speed.mean()), p95_speed_mps=float(np.percentile(speed, 95)),
                  observed_distance_m=distance, distance_censored=len(observed) != raw_steps,
                  stopped_seconds=float(np.sum(speed < .1) * .1), stopped_observed_fraction=float(np.mean(speed < .1)),
                  longitudinal_accel_rms_mps2=rms(accel), longitudinal_accel_abs_p95_mps2=abs95(accel),
                  jerk_rms_mps3=rms(jerk), jerk_abs_p95_mps3=abs95(jerk),
                  jerk_sample_count=len(jerk), hard_braking_observed_fraction=float(np.mean(accel < -3.)),
                  hard_acceleration_observed_fraction=float(np.mean(accel > 3.)),
                  high_jerk_observed_fraction=float(np.mean(np.abs(jerk) > 5.)) if jerk else None,
                  lateral_accel_proxy_rms_mps2=rms(lateral), lateral_accel_proxy_abs_p95_mps2=abs95(lateral),
                  mean_abs_lateral_speed_mps=float(np.mean([abs(s['lateral_speed']) for s in observed])),
                  leader_observed_fraction=leader_count / len(observed),
                  closing_leader_sample_count=len(ttc), minimum_following_ttc_s=min(ttc) if ttc else None,
                  following_ttc_below_1_5s_observed_fraction=sum(t < 1.5 for t in ttc) / len(observed),
                  following_ttc_below_3s_observed_fraction=sum(t < 3 for t in ttc) / len(observed),
                  following_ttc_below_1_5s_conditional_fraction=float(np.mean(np.array(ttc) < 1.5)) if ttc else None,
                  minimum_following_headway_s=min(headway) if headway else None,
                  allowed_speed_excess_fraction=float(np.mean([s['speed'] > s['allowed_speed'] + .1 for s in observed])),
                  observed_lane_changes=lane_changes, infeasible_command_fraction=1 - float(np.mean(feas)) if feas else None,
                  action_mask_coverage=len(feas) / len(actions) if actions else None)
    return result


def aggregate_driving(episodes):
    keys = sorted(set().union(*(e['driving'].keys() for e in episodes)))
    metrics = {key: distribution([e['driving'][key] for e in episodes if e['driving'].get(key) is not None]) for key in keys}
    distance = sum(e['driving']['observed_distance_m'] for e in episodes)
    metrics['collision_episodes_per_observed_km'] = sum(e['record']['collision'] for e in episodes) / (distance / 1000) if distance else None
    metrics['total_observed_distance_m'] = distance
    metrics['aggregation'] = 'Each scalar: equal-episode mean/population std, n excludes missing; distance and collision/km pooled.'
    return metrics
