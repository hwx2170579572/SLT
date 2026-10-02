"""Read-only experiment audit, with a separate completion evidence receipt."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import time

import numpy as np

from tools.phase1_checkpoint_diagnostics import validate_report, summarize
from tools.phase3_mechanism_v1.common import ROOT, OUT, read, sha, write, digest
from tools.phase3_mechanism_v1.training import check_preflight
from tools.phase3_mechanism_v1.evaluation import selection_key


def audit():
    check_preflight()
    jobs = read(OUT / 'queue_manifest.json')
    receipts = {p.stem: read(p) for p in (OUT / 'jobs').glob('*.json')}
    assert len(jobs) == 450 and set(receipts) == set(jobs)
    hashed, evaluated, origins = {}, {}, Counter()
    traffic_groups, overlaps, train_count, gradient_count, selection_count = {}, [], 0, 0, 0
    episode_count = 0

    def checked(path, expected):
        path = (ROOT / path).resolve()
        key = str(path)
        if key not in hashed:
            hashed[key] = sha(path)
        assert hashed[key] == expected, f'Hash changed: {path}'

    for key, receipt in receipts.items():
        job = jobs[key]
        assert receipt['job'] == job
        checked(receipt['artifact'], receipt['artifact_sha256'])
        artifact = ROOT / receipt['artifact']
        result = read(artifact)
        if job['kind'] == 'train':
            train_count += 1
            assert not result['smoke'] and result['raw_steps'] == 50000 and result['updates'] == 45001
            assert result['tensor_roundtrip_equal']
            checked(artifact.parent / 'final_model.zip', result['checkpoint_sha256'])
            for path, expected in result['checkpoints'].items():
                checked(path, expected)
        elif job['kind'] == 'eval':
            if receipt['artifact'] in evaluated:
                continue
            evaluated[receipt['artifact']] = result
            identity = result['identity']
            assert not identity['smoke']
            validate_report(result, identity['episodes'], identity['seed_start'])
            recalculated = summarize(result['episode_records'])
            for field, value in recalculated['summary'].items():
                assert np.isclose(value, result['summary'][field], atol=1e-10, rtol=0)
            checked(result['checkpoint']['path'], result['checkpoint']['sha256'])
            origins[result['origin']] += 1
            if result['origin'] == 'new_real_rollouts':
                assert result['tensor_state_unchanged']
                episode_count += len(result['episode_records'])
                for offset, record in enumerate(result['episode_records']):
                    trace = read(artifact.parent / f'episode_{offset:03d}.json')
                    assert trace['record'] == record and trace['identity_sha256'] == digest(identity)
                    flags = [record[x] for x in ('success', 'collision', 'off_route', 'timeout')]
                    if sum(flags) > 1:
                        overlaps.append({'artifact': receipt['artifact'], 'seed': record['seed'], 'flags': flags})
                    if 'bank' in trace:
                        checked(trace['bank']['path'], trace['bank']['sha256'])
                    traffic = trace['traffic']
                    if traffic:
                        files = traffic['files']
                        for f in files:
                            checked(f['path'], f['sha256'])
                        signature = ([f['sha256'] for f in files], traffic['ego_routes_sha256'], traffic['network_sha256'])
                        group = (identity['scenario'], identity['mode'], record['seed'])
                        if group in traffic_groups:
                            assert traffic_groups[group] == signature, f'Paired traffic differs: {group}'
                        else:
                            traffic_groups[group] = signature
            else:
                assert result['origin'] == 'reused_completed_evaluation'
                prior = read(ROOT / result['source'])
                assert result['episode_records'] == prior['episode_records']
        elif job['kind'] == 'select':
            selection_count += 1
            candidates = []
            for c in result['candidate_results']:
                checked(c['path'], c['sha256'])
                candidate = read(ROOT / c['path'])
                validate_report(candidate, 100, 310000)
                candidates.append(candidate)
            assert sorted(r['checkpoint']['raw_steps'] for r in candidates) == [10000, 20000, 30000, 40000, 50000]
            assert max(candidates, key=selection_key)['checkpoint'] == result['selected_checkpoint']
            assert result['scoring_outcomes_used'] is False
        elif job['kind'] == 'gradient':
            gradient_count += 1
            assert not result['identity']['smoke'] and not result['training_performed']
            assert result['tensor_state_unchanged'] and not result['bank_is_original_training_replay']
            checked(result['identity']['bank_result'], result['identity']['bank_result_sha256'])
            for field, aggregate in result['aggregates'].items():
                values = [r[field] for r in result['rows'] if r[field] is not None]
                if values:
                    assert np.isclose(np.mean(values), aggregate['mean'], atol=1e-10, rtol=0)
            if result['actual_isolation']:
                assert result['aggregates']['actual_lane_trunk_gradient_norm']['max'] == 0

    recovery = read(OUT / 'recovery_20260916_v1/audit.json')
    for name, expected in recovery['completed_receipts_sha256'].items():
        checked(OUT / 'jobs' / name, expected)
    for path, expected in recovery['existing_episode_sha256'].items():
        checked(path, expected)
    assert (train_count, gradient_count, selection_count) == (12, 45, 30)
    for key, receipt in receipts.items():
        job = jobs[key]
        if job['kind'] == 'eval' and job['mode'] == 'score' and job['dependencies']:
            selection_receipt = receipts[job['dependencies'][0]]
            if selection_receipt['job']['kind'] == 'select':
                selected = read(ROOT / selection_receipt['artifact'])
                assert selected['sealed_at_unix'] <= receipt['completed_at']

    curves = ROOT / read(OUT / 'figures_v1/latest.json')['snapshot']
    curve_receipt = read(curves / 'completion.json')
    for path, expected in curve_receipt['artifacts'].items():
        checked(curves / path, expected)
    assert curve_receipt['new_training_complete'] == 12
    assert not read(curves / 'source_manifest.json')['missing_training']

    payload = {'passed': True, 'audited_at': time.time(), 'completed_jobs': len(receipts),
               'job_kinds': dict(Counter(j['kind'] for j in jobs.values())),
               'unique_evaluation_artifacts': len(evaluated), 'evaluation_origins': dict(origins),
               'new_rollout_episodes': episode_count, 'paired_fresh_traffic_groups': len(traffic_groups),
               'verified_unique_file_hashes': len(hashed),
               'pre_recovery_receipts_unchanged': len(recovery['completed_receipts_sha256']),
               'pre_recovery_episodes_unchanged': len(recovery['existing_episode_sha256']),
               'overlapping_outcome_flags': overlaps,
               'outcome_note': 'Collision/off-route/timeout flags are not necessarily mutually exclusive; do not force their sum to 100%.',
               'plots_snapshot': curves.relative_to(ROOT).as_posix(),
               'script_sha256': sha(Path(__file__)), 'execution_preflight_sha256': sha(OUT / 'preflight.json'),
               'multiseed_confirmation_deferred': True, 'formal_test_accessed': False}
    write(OUT / 'completion_audit_v1.json', payload)
    return {k: v for k, v in payload.items() if k != 'overlapping_outcome_flags'} | {'overlapping_outcome_episodes': len(overlaps)}


if __name__ == '__main__':
    print(json.dumps(audit(), ensure_ascii=False))
