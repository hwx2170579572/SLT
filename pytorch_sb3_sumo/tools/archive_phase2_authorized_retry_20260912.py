"""One-time preservation for the four retries explicitly authorized by the user."""
import hashlib
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.control_phase2_gpu_v7 import live_workers
from tools.phase1_checkpoint_diagnostics import write, sha


def main():
    if live_workers():
        raise RuntimeError('Workers still alive')
    ids = ['v4_13__left_turn__tau_double__seed0', 'v4_13__roundabout__lr_half__seed0',
           'v4_8__cross__tau_double__seed0', 'v4_8__left_turn__tau_double__seed0']
    archive = ROOT/'results_phase2_runtime_v2/interrupted_archive_20260912_retry1'
    if archive.exists():
        raise RuntimeError('Archive exists; inspect before continuing')
    moves = []
    for job in ids:
        cell = ROOT/'results_phase2_runtime_v2/screen'/job
        if not cell.is_dir() or (cell/'training_complete.json').exists():
            raise RuntimeError('Unexpected training state: '+job)
        namespace = hashlib.sha256(str(cell).encode()).hexdigest()[:12]
        relative = [f'results_phase2_runtime_v2/screen/{job}',
                    f'results_phase2_runtime_v2/tb/{namespace}', f'results_phase2_runtime_v2/ov/{namespace}',
                    f'results_phase2_runtime_v2/logs/{job}.process.json',
                    f'results_phase2_reuse_v3/logs/{job}.stdout.log', f'results_phase2_reuse_v3/logs/{job}.stderr.log']
        for rel in relative:
            src, dst = (ROOT/rel).resolve(), (archive/rel).resolve()
            if not src.is_relative_to(ROOT) or not dst.is_relative_to(archive):
                raise ValueError('Path escapes workspace')
            if src.exists():
                moves.append((src, dst))
    archive.mkdir()
    write(archive/'authorization_and_moves.json', dict(authorization='User: 这四个单元全部重跑',
        jobs=ids, workers=3, device='cuda', moves=[dict(source=str(s), destination=str(d)) for s,d in moves]))
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
    write(archive/'preservation_complete.json', dict(moved=len(moves),
        original_status_hashes={job:sha(archive/f'results_phase2_runtime_v2/screen/{job}/status.json') for job in ids}))
    print('Preserved', len(moves), 'paths')


if __name__ == '__main__':
    main()
