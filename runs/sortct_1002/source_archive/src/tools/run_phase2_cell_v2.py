"""Independent phase-2 cell runner; smoke outputs never enter scientific ranking."""
from __future__ import annotations
import argparse
import sys
import time
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.phase1_checkpoint_diagnostics import read, write, sha, validate_report
from tools.phase2_model_factory_v1 import make_phase2_model, verify_optimizer_settings


def run(args):
    import torch
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from algos.sb3_torch import RawStepControlCallback, evaluate_model_detailed
    from tools.train_independent_v2_5m6s100e_v1 import _make_environment_factory
    from tools.paper_evaluation_contract import validate_model_environment_spaces
    torch.set_num_threads(1)
    contract_path = ROOT/'results_phase2_diagnosis_20260908/tuning_contract_v2.json'
    contract = read(contract_path)
    if args.method not in contract['methods'] or args.scenario not in contract['scenarios']:
        raise ValueError('Method/scenario outside frozen phase-2 scope')
    candidate = next(c for c in contract['candidates'] if c['id']==args.candidate)
    if args.method == contract['baseline_method'] and args.candidate not in contract['baseline_candidates']:
        raise ValueError('MST+SLT is the fixed control in the v2 optimization round')
    if args.seed not in contract['screen_training_seeds']:
        raise ValueError('Confirmation seeds require a separate selected-candidate manifest')
    if not args.smoke:
        receipt = read(ROOT/'results_phase2_runtime_v2/preflight_v2.json')
        if not receipt['passed'] or receipt['contract_sha256'] != sha(contract_path):
            raise ValueError('Missing or stale phase-2 preflight')
        for path, expected in receipt['source_sha256'].items():
            if sha(ROOT/path) != expected:
                raise ValueError(f'Implementation changed after preflight: {path}')
    output = ROOT/'results_phase2_runtime_v2'/('smoke' if args.smoke else 'screen')/f'{args.method}__{args.scenario}__{args.candidate}__seed{args.seed}'
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite an interrupted or completed run.
    started = time.time()
    write(output/'status.json', dict(status='starting', smoke=args.smoke, pid=__import__('os').getpid()))
    plan = read(ROOT/'results_phase1_checkpoint_diagnostics_v1/plan.json')
    source = next(s for s in plan['sources'] if s['method']==args.method and s['scenario']==args.scenario)
    requested = read(ROOT/source['run']/'arguments.json')['requested_raw_steps']
    env_args = argparse.Namespace(**dict(requested, seed=args.seed, gui=False, evaluation_split='validation'))
    protocol = read(ROOT/plan['protocol'])
    namespace = hashlib.sha256(str(output).encode()).hexdigest()[:12]
    factory = _make_environment_factory(adapter='base' if args.method in ('mst_slt','temporal_graph') else args.method,
        density=protocol['scenarios'][args.scenario], overlay_root=ROOT/'results_phase2_runtime_v2'/'ov'/namespace)
    raw_budget, warmup, batch, buffer = (72, 48, 2, 128) if args.smoke else (50000, 5000, 32, 20000)
    lr, tau = candidate['learning_rate'], candidate['tau']
    env = None; evaluation_env = None
    checks = []
    class OptimizerAudit(BaseCallback):
        def _on_step(self):
            verify_optimizer_settings(self.model, learning_rate=lr, tau=tau)
            if self.n_calls % 100 == 0:
                write(output/'progress.json', dict(raw_steps=self.model._raw_steps_seen,
                    learner_updates=self.model._n_updates, updated_at=time.time()))
            return True
        def _on_training_end(self):
            checks.append(verify_optimizer_settings(self.model, learning_rate=lr, tau=tau))
    try:
        env = Monitor(factory(env_args), filename=str(output/'train_monitor.csv'),
            info_keywords=('raw_simulation_steps','is_success','collision','off_route','max_time'))
        model = make_phase2_model(args.method, env, learning_rate=lr, tau=tau, scenario=args.scenario,
            batch_size=batch, learning_starts=warmup, buffer_size=buffer, action_repeat=3, seed=args.seed,
            device=args.device, verbose=0,
            tensorboard_log=str(ROOT/'results_phase2_runtime_v2'/'tb'/namespace))
        write(output/'arguments.json', dict(method=args.method, scenario=args.scenario, candidate=candidate,
            smoke=args.smoke, raw_budget=raw_budget, warmup=warmup, batch_size=batch, buffer_size=buffer,
            seed=args.seed, decoder=source['decoder'], contract_sha256=sha(contract_path),
            runner_sha256=sha(Path(__file__)), model_factory_sha256=sha(ROOT/'tools/phase2_model_factory_v1.py'),
            environment_protocol_sha256=sha(ROOT/plan['protocol']), inherited_environment_arguments=vars(env_args)))
        model.learn(total_timesteps=raw_budget, callback=CallbackList([
            RawStepControlCallback(raw_step_budget=raw_budget, checkpoint_frequency=36 if args.smoke else 10000,
                checkpoint_path=output/'checkpoints', checkpoint_prefix='ckpt'), OptimizerAudit()]))
        assert model._raw_steps_seen == raw_budget
        assert model._n_updates == raw_budget-warmup+1
        verify_optimizer_settings(model, learning_rate=lr, tau=tau)
        checkpoint = output/'final_model.zip'; model.save(checkpoint)
        write(output/'training_diagnostics.json', model.training_diagnostics())
        write(output/'training_complete.json', dict(checkpoint_sha256=sha(checkpoint),
            raw_steps=model._raw_steps_seen, learner_updates=model._n_updates,
            optimizer_settings=verify_optimizer_settings(model, learning_rate=lr, tau=tau)))
        evaluation_env = factory(env_args, evaluation=True)
        if args.method == 'v4_13':
            from tools.action_diagnostics_v4_13_model import load_model_for_deployment_v4_13
            restored = load_model_for_deployment_v4_13(type(model), checkpoint, decoder=source['decoder'], env=evaluation_env, device=args.device)
        elif args.method == 'v4_8':
            from tools.action_diagnostics_v4_6 import load_model_for_deployment
            restored = load_model_for_deployment(type(model), checkpoint, decoder=source['decoder'], env=evaluation_env, device=args.device)
        else:
            restored = type(model).load(checkpoint, env=evaluation_env, device=args.device)
        after = verify_optimizer_settings(restored, learning_rate=lr, tau=tau)
        assert restored._n_updates == model._n_updates
        assert restored._raw_steps_seen == model._raw_steps_seen
        for key, tensor in model.policy.state_dict().items():
            assert torch.equal(tensor.cpu(), restored.policy.state_dict()[key].cpu()), key
        validate_model_environment_spaces(restored, evaluation_env)
        from tools.phase1_checkpoint_diagnostics import summarize
        episodes = 1 if args.smoke else 100
        records = []
        reference = read(ROOT/source['run']/'paper_evaluation_detailed.json')['episode_records']
        for offset in range(0, episodes, 20):
            count = min(20, episodes-offset)
            d = evaluate_model_detailed(restored, evaluation_env, episodes=count, seed=10000+offset,
                deterministic=True, sumo_step_seconds=.1, policy_action_hold=1).to_dict()
            validate_report(d, count, 10000+offset)
            assert [r['traffic_variant'] for r in d['episode_records']] == [r['traffic_variant'] for r in reference[offset:offset+count]]
            for r in d['episode_records']: r['episode'] = r['seed']-10000
            records.extend(d['episode_records'])
            write(output/f'evaluation_block_{offset:03d}.json', dict(**d, checkpoint_sha256=sha(checkpoint), decoder=source['decoder'], smoke=args.smoke))
        write(output/'evaluation.json', dict(**summarize(records), checkpoint_sha256=sha(checkpoint), decoder=source['decoder'], smoke=args.smoke))
        write(output/'training_diagnostics.json', model.training_diagnostics())
        write(output/'status.json', dict(status='completed', smoke=args.smoke, scientific_result=not args.smoke,
            raw_steps=model._raw_steps_seen, learner_updates=model._n_updates, optimizer_checks=checks,
            restored_optimizer_settings=after, policy_tensor_roundtrip_equal=True,
            evaluation_episodes=episodes, wall_seconds=time.time()-started))
        print(str(output), flush=True)
    except BaseException as exc:
        write(output/'status.json', dict(status='failed', smoke=args.smoke, error=repr(exc), wall_seconds=time.time()-started))
        raise
    finally:
        if evaluation_env is not None: evaluation_env.close()
        if env is not None: env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True)
    parser.add_argument('--scenario', default='cross')
    parser.add_argument('--candidate', default='control')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())
