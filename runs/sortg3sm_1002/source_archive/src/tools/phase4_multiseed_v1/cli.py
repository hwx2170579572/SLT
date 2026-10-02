from __future__ import annotations

import argparse
import json

from .common import source_registry


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('command', choices=['prepare', 'train', 'eval', 'run', 'analyze'])
    p.add_argument('--source')
    p.add_argument('--step', default='50000')
    p.add_argument('--mode', default='selection')
    p.add_argument('--decoder', default='native')
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.command == 'prepare':
        from .plan import prepare
        data = prepare()
        result = {'cells': len(data['sources']), 'new_training_jobs': len(data['training_jobs']),
                  'reused_seed_zero_cells': data['matrix']['reused_seed_zero_cells']}
    elif args.command == 'train':
        from .training import train
        result = train(args.source, args.device, args.smoke)
    elif args.command == 'eval':
        source = source_registry()[args.source]
        from .evaluation import evaluate
        result = evaluate(source, args.step, args.mode, args.decoder, args.device, args.smoke)
    elif args.command == 'run':
        from .controller import run
        result = run()
    else:
        from .analyze import analyze
        result = analyze()
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
