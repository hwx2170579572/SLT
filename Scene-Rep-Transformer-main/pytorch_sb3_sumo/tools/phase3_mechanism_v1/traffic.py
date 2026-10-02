"""New traffic realizations for selection/scoring; legacy training is unchanged.

Only social-traffic departures change. Routes, ego traffic, vehicle types,
observations, rewards and episode limits use the frozen environment. CARLA
shares its only route template across splits; this is disclosed explicitly.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from envs.sumo.independent_v2_five_methods_six_scenarios_100ep_v1 import (
    IndependentV2FiveBySixEnvV1, IndependentV2FiveBySixEnvV4V1,
)
from .common import OUT, PROTOCOL, ROOT, digest, read, seal, sha

MODES = {
    'selection': (310000, 100, 1.),
    'score': (320000, 100, 1.),
    'robust_lower': (330000, 100, .9),
    'robust_higher': (340000, 100, 1.1),
    'mechanism': (350000, 32, 1.),
}


def jitter_document(raw: bytes, seed: int, width: float = .75):
    tree = ET.fromstring(raw)
    rng = random.Random(seed)
    changed = 0
    static, timed = [], []
    for node in tree:
        if node.tag not in ('vehicle', 'trip', 'person', 'flow', 'personFlow'):
            static.append(node)
            continue
        attribute = 'begin' if node.tag in ('flow', 'personFlow') else 'depart'
        try:
            original = float(node.attrib[attribute])
        except (KeyError, ValueError):
            static.append(node)
            continue
        shifted = max(0., original + rng.uniform(-width, width))
        delta = shifted - original
        node.set(attribute, f'{shifted:.6f}')
        if attribute == 'begin' and 'end' in node.attrib:
            node.set('end', f'{float(node.attrib["end"]) + delta:.6f}')
        timed.append((shifted, node))
        changed += 1
    tree[:] = static + [node for _, node in sorted(timed, key=lambda item: item[0])]
    if not changed:
        raise ValueError('No numeric social departures to randomize')
    return ET.tostring(tree, encoding='utf-8', xml_declaration=True), changed


class FreshTrafficMixin:
    def __init__(self, *args, phase3_mode, **kwargs):
        if phase3_mode not in MODES:
            raise ValueError(phase3_mode)
        self.phase3_mode = phase3_mode
        self.phase3_traffic_receipt = None
        super().__init__(*args, **kwargs)

    def _sumo_command(self, seed):
        # Absolute seed determines template, so resume/chunk order cannot change traffic.
        self._traffic_episode_index = int(seed)
        self._traffic_roll = None
        command = super()._sumo_command(seed)
        index = command.index('--route-files') + 1
        originals = [Path(p) for p in command[index].split(',')]
        if len(originals) != 3:
            raise ValueError('Expected source social traffic, additive overlay, ego routes')
        receipts, replacements = [], []
        for position, original in enumerate(originals[:-1]):
            original_sha = sha(original)
            key = {'mode': self.phase3_mode, 'simulation_seed': int(seed),
                   'source_sha256': original_sha, 'position': position, 'jitter_seconds': .75}
            token = digest(key)
            dest = OUT / 'traffic' / self.scenario / self.phase3_mode / f'{token[:16]}.rou.xml'
            if not dest.exists():
                raw, changed = jitter_document(original.read_bytes(), int(token[:16], 16))
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Distinct jobs may request identical paired traffic simultaneously.
                from .common import write
                import os
                temp = dest.with_name(dest.name + f'.{os.getpid()}.tmp')
                temp.write_bytes(raw)
                temp.replace(dest)
            else:
                raw, changed = jitter_document(original.read_bytes(), int(token[:16], 16))
                if hashlib.sha256(raw).hexdigest() != sha(dest):
                    raise ValueError('Generated traffic content drift')
            receipt = {**key, 'path': str(dest), 'sha256': sha(dest), 'changed_departures': changed}
            if receipt['sha256'] == original_sha:
                raise ValueError('Fresh traffic matches training realization')
            receipts.append(receipt)
            replacements.append(str(dest))
        command[index] = ','.join(replacements + [str(originals[-1])])
        self.phase3_traffic_receipt = {
            'mode': self.phase3_mode, 'seed': int(seed), 'files': receipts,
            'template': self._selected_traffic_path.name,
            'template_shared_with_training': len(self._paper_specification.traffic_paths) < 2,
            'new_realizations': True, 'ego_routes_sha256': sha(originals[-1]),
            'network_sha256': sha(self._paper_specification.network_path),
        }
        return command


class Phase3BaseEnv(FreshTrafficMixin, IndependentV2FiveBySixEnvV1):
    pass


class Phase3V4Env(FreshTrafficMixin, IndependentV2FiveBySixEnvV4V1):
    pass


def requested_arguments(source):
    record = read(ROOT / source['arguments'])
    return record.get('requested_raw_steps', record.get('inherited_environment_arguments', record))


def make_env(source, mode, namespace, training=False):
    args = argparse.Namespace(**copy.deepcopy(requested_arguments(source)))
    args.gui = False
    args.seed = 0
    args.evaluation_split = 'train' if mode == 'legacy_train' else 'validation'
    density = copy.deepcopy(read(PROTOCOL)['scenarios'][source['scenario']])
    adapter = 'base' if source['method'] == 'mst_slt' else source['method']
    if mode in ('legacy', 'legacy_train') or training:
        from tools.train_independent_v2_5m6s100e_v1 import _make_environment_factory
        factory = _make_environment_factory(adapter=adapter, density=density,
                                            overlay_root=OUT / 'ov' / namespace)
        return factory(args, evaluation=not training)
    multiplier = MODES[mode][2]
    density['vehicle_scale'] *= multiplier
    if density['pedestrian_scale'] > 1.:
        density['pedestrian_scale'] *= multiplier
    cls = Phase3BaseEnv if adapter == 'base' else Phase3V4Env
    return cls(
        phase3_mode=mode, scenario=args.scenario, history_steps=args.history_steps,
        neighbors=args.neighbors, path_length=args.path_length,
        action_repeat=args.action_repeat, reward_discount=args.discount,
        ego_control_profile=args.ego_control_profile, include_state_lstm=False,
        state_lstm_only=False, episode_limit_profile=args.episode_limit_profile,
        render_mode=None, high_density_vehicle_scale=density['vehicle_scale'],
        high_density_pedestrian_scale=density['pedestrian_scale'],
        high_density_clone_jitter_seconds=tuple(density['clone_depart_jitter_seconds']),
        high_density_overlay_root=OUT / 'ov' / namespace,
        high_density_partition='evaluation',
        high_density_contract_partition='evaluation' if adapter == 'base' else 'validation',
    )
