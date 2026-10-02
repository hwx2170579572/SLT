"""为之前汇总遗漏的实验结果目录补充提取 raw JSON。

仅提取「有真实评估指标（success_rate / collision_rate / timeout_rate / mean_return）」
的目录；无指标的过程性目录只生成一个空的 document（rows=[]），由 build_excel.py
作为批次索引记录，绝不强行捏造指标行。冒烟实验（smoke / preflight）一律跳过。

每个产出 JSON 的结构与既有 raw/*.json 完全一致：
  { directory, phase, experiment_stage, purpose, status, protocol_files, rows, missing, notes }
"""
from __future__ import annotations

import csv
import json
import os
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / 'raw'
ROOT = HERE.parent.parent  # pytorch_sb3_sumo 目录


def jload(path: pathlib.Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:  # 读取失败必须可见，不静默跳过
        return {'__error__': str(exc)}


def _first(*vals):
    for v in vals:
        if v is not None and v != '':
            return v
    return None


def per_run_to_row(directory, phase, stage, purpose, evidence_kind, pr, source_file, status):
    """把 summary 的一条 per_run 记录映射为与 build_excel.py 一致的 row。"""
    method = pr.get('method')
    scenario = pr.get('scenario')
    seed = pr.get('seed')

    # 晋升线里 temporal_graph 是统一控制基线（baseline），其余方法为候选。
    if evidence_kind == 'promotion':
        if method == 'temporal_graph':
            variant, candidate = 'temporal_graph', 'control'
        else:
            variant, candidate = method, 'candidate'
    elif evidence_kind == 'ablation':
        variant, candidate = method, 'ablation'
    else:
        variant, candidate = method, None

    success = pr.get('success_rate')
    return {
        'batch': directory,
        'phase': phase,
        'variant': variant,
        'candidate': candidate,
        'scenario': scenario,
        'training_seed': seed,
        'model_kind': pr.get('kind'),
        'checkpoint_step': pr.get('raw_steps'),
        'episodes': pr.get('evaluation_episodes'),
        'eval_seed_start': _first(pr.get('evaluation_seed_start'), pr.get('eval_seed_start')),
        'decoder': _first(pr.get('selected_deployment_decoder'), pr.get('deployment_decoder')),
        'success_rate': success,
        'collision_rate': pr.get('collision_rate'),
        'timeout_rate': pr.get('timeout_rate'),
        'mean_return': pr.get('mean_return'),
        'status': status,
        'evidence_kind': evidence_kind,
        'artifact': source_file,
        'notes': _first(pr.get('algorithm'), pr.get('run_name')),
        'source_keys': {
            'success_rate': 'summary.per_run[].success_rate',
            'collision_rate': 'summary.per_run[].collision_rate',
            'timeout_rate': 'summary.per_run[].timeout_rate',
            'mean_return': 'summary.per_run[].mean_return',
            'episodes': 'summary.per_run[].evaluation_episodes',
            'eval_seed_start': 'summary.per_run[].evaluation_seed_start',
            'checkpoint_step': 'summary.per_run[].raw_steps',
            'decoder': 'summary.per_run[].selected_deployment_decoder',
            'variant': 'summary.per_run[].method',
        },
    }


def extract_summary(directory, phase, stage, purpose, sources):
    """sources: list[(相对路径, evidence_kind)]，逐个读取 summary.json 的 per_run。"""
    rows = []
    missing = []
    protocol_files = []
    statuses = []
    for rel, evidence_kind in sources:
        path = ROOT / directory / rel
        if not path.exists():
            missing.append({'unit': rel, 'reason': f'缺少文件 {rel}'})
            continue
        doc = jload(path)
        if isinstance(doc, dict) and '__error__' in doc:
            missing.append({'unit': rel, 'reason': doc['__error__']})
            continue
        if isinstance(doc, list):
            # per_run.json 直接是数组
            per_run = doc
            complete = None
            rejected = []
        elif isinstance(doc, dict):
            per_run = doc.get('per_run') or []
            complete = doc.get('complete')
            rejected = []
        else:
            missing.append({'unit': rel, 'reason': '非对象结构'})
            continue
        protocol_files.append(f'{directory}/{rel}')
        for pr in per_run:
            status = 'completed' if pr.get('success_rate') is not None else 'partial'
            rows.append(per_run_to_row(directory, phase, stage, purpose, evidence_kind,
                                       pr, f'{directory}/{rel}', status))
        # 未完成 / 被拒 / 缺失单元，记录到 missing
        if isinstance(doc, dict):
            for key in ('rejected_or_missing', 'missing_jobs', 'unresolved'):
                for item in doc.get(key) or []:
                    missing.append({'unit': item if isinstance(item, str) else json.dumps(item, ensure_ascii=False),
                                    'reason': key})
        statuses.append(complete)
    overall = 'completed' if statuses and all(s is True for s in statuses) else (
        'partial' if any(s is True for s in statuses) or rows else 'empty')
    return {
        'directory': directory,
        'phase': phase,
        'experiment_stage': stage,
        'purpose': purpose,
        'status': overall,
        'protocol_files': protocol_files,
        'rows': rows,
        'missing': missing,
        'notes': None,
    }


def extract_phase2_report():
    directory = 'results_phase2_systematic_report_v1'
    phase = 'phase2 系统报告（三方法 × 参数候选 × 六场景 100 回合）'
    stage = 'phase1~4 确认线'
    purpose = ('phase2 系统性报告：mst_slt / v4_8 / v4_13 三种方法，各含 control、lr_half、lr_quarter、'
               'tau_half、tau_double 五种参数候选，在 left_turn/cross/roundabout_easy/roundabout_medium/'
               'roundabout/carla 六场景 × seed0 下各评 100 回合，形成 all_cells.csv 结果矩阵。')
    rows = []
    path = ROOT / directory / 'all_cells.csv'
    missing = []
    if not path.exists():
        missing.append({'unit': 'all_cells.csv', 'reason': '缺少文件'})
    else:
        text = path.read_text(encoding='utf-8-sig')
        reader = csv.DictReader(text.splitlines())
        for rec in reader:
            def fnum(k):
                try:
                    return float(rec[k])
                except (ValueError, TypeError):
                    return None
            rows.append({
                'batch': directory,
                'phase': phase,
                'variant': rec.get('method'),
                'candidate': rec.get('candidate'),
                'scenario': rec.get('scenario'),
                'training_seed': _first(rec.get('seed'), None),
                'model_kind': None,
                'checkpoint_step': None,
                'episodes': _first(rec.get('episodes'), None),
                'eval_seed_start': None,
                'decoder': None,
                'success_rate': fnum('success_rate'),
                'collision_rate': fnum('collision_rate'),
                'timeout_rate': fnum('timeout_rate'),
                'mean_return': fnum('mean_return'),
                'status': 'completed',
                'evidence_kind': 'systematic_report',
                'artifact': f'{directory}/all_cells.csv',
                'notes': 'all_cells.csv 结果矩阵行',
                'source_keys': {
                    'success_rate': 'all_cells.csv[success_rate]',
                    'collision_rate': 'all_cells.csv[collision_rate]',
                    'timeout_rate': 'all_cells.csv[timeout_rate]',
                    'mean_return': 'all_cells.csv[mean_return]',
                },
            })
    return {
        'directory': directory,
        'phase': phase,
        'experiment_stage': stage,
        'purpose': purpose,
        'status': 'completed' if rows else 'empty',
        'protocol_files': [f'{directory}/all_cells.csv'] if path.exists() else [],
        'rows': rows,
        'missing': missing,
        'notes': None,
    }


def empty_document(directory, phase, stage, purpose, status, note):
    """无真实指标的目录：仅批次索引，rows=[]。"""
    return {
        'directory': directory,
        'phase': phase,
        'experiment_stage': stage,
        'purpose': purpose,
        'status': status,
        'protocol_files': [],
        'rows': [],
        'missing': [],
        'notes': note,
    }


def main():
    os.makedirs(RAW, exist_ok=True)

    dev_sources = {
        'results_topo_v4_3_dev': [('development/summary.json', 'development')],
        'results_topo_v4_7_2_dev': [('development/summary.json', 'development')],
        'results_topo_v4_8_dev': [('development/summary.json', 'development')],
        'results_topo_v4_9_dev': [('development/summary.json', 'development')],
        'results_topo_v4_10_dev': [('development/summary.json', 'development')],
        'results_topo_v4_11_dev': [('development/summary.json', 'development'),
                                   ('ablation/summary.json', 'ablation')],
        'results_topo_v4_12_dev': [('development/summary.json', 'development')],
        'results_topo_v4_13_dev': [('development/summary.json', 'development'),
                                   ('ablation/summary.json', 'ablation')],
    }
    dev_phases = {
        'results_topo_v4_3_dev': 'topo v4.3 开发线（Target-Critic Lane Decoder，目标车道解码器）',
        'results_topo_v4_7_2_dev': 'topo v4.7.2 工程补丁线（Parent-Preregistration Context Isolation Patch）',
        'results_topo_v4_8_dev': 'topo v4.8 开发线（Tie-Only Replicated Train Calibration）',
        'results_topo_v4_9_dev': 'topo v4.9 开发线（Learned Collision-Value Model）',
        'results_topo_v4_10_dev': 'topo v4.10 开发线（Shared-Supported Mixture Model，共享风险支持混合模型）',
        'results_topo_v4_11_dev': 'topo v4.11 开发线（Proper-Calibrated Ranked Risk Model，PRCR）',
        'results_topo_v4_12_dev': 'topo v4.12 开发线（Augmented Joint-Support PRCR）',
        'results_topo_v4_13_dev': 'topo v4.13 开发线（Gradient-Isolated Tempered Joint Support PRCR）',
    }
    prom_sources = {
        'results_topo_v4_8_promotion': ('summary.json', 'topo v4.8 晋升线（正式配对对照）'),
        'results_topo_v4_9_2_promotion': ('engineering_recovery_v4_9_2_3/summary.json',
                                          'topo v4.9.2 晋升线（严格纯学习部署协议，正式配对对照）'),
        'results_topo_v4_11_promotion': ('summary.json', 'topo v4.11 晋升线（正式配对对照）'),
        'results_topo_v4_13_promotion': ('summary.json', 'topo v4.13 晋升线（正式配对对照）'),
    }

    documents = []

    for directory, sources in dev_sources.items():
        documents.append(extract_summary(
            directory, dev_phases[directory], 'topo v4 版本线',
            f'{dev_phases[directory]} 的 development/ablation 阶段逐运行结果。', sources))

    for directory, (rel, phase) in prom_sources.items():
        documents.append(extract_summary(
            directory, phase, 'topo v4 版本线',
            f'{phase} 的正式配对训练结果（候选方法 vs temporal_graph 控制基线）。',
            [(rel, 'promotion')]))

    documents.append(extract_phase2_report())

    # 无真实指标 / 未完成的过程性目录，仅作为批次索引记录
    documents.append(empty_document(
        'results_high_density_same_scene_v1', '高密度同场景多方法对比线（comparison，未跑完）', '高密度对比',
        '同场景高密度对比协议（54 个单元）。comparison 阶段 accepted_jobs=0、per_run 为空，未产生真实评估指标。',
        'empty', 'comparison/summary.json 显示 complete=false，无逐运行结果。'))
    documents.append(empty_document(
        'results_topo_v4_9_2_dev', 'topo v4.9.2 开发线（严格纯学习部署协议，仅工程补丁）', 'topo v4 版本线',
        'v4.9.2 开发目录仅含工程补丁与冒烟验证（engineering/smoke），无 development 逐运行科学结果。',
        'empty', '无 development/summary.json，仅 engineering 补丁与 smoke。'))
    documents.append(empty_document(
        'results_topo_v4_10', 'topo v4.10 运行前检查线（单个 check_result）', 'topo v4 版本线',
        '仅含一次 left_turn seed0 的运行前 check_result（动作有限性/模型类检查），无完整训练评估。',
        'empty', 'check_result.json 无 success_rate 等评估指标。'))
    documents.append(empty_document(
        'results_visual_eval_three_methods_20ep_v1', '三方法六场景视觉评估线（20 回合，失败）', '独立v2',
        'mst_slt_default / v4_8_lr_half / v4_13_tau_half 三方法六场景各 20 回合视觉评估；manifest 显示 status=failed、completed_cells=[]，因缺 MST+SLT exact_final 源而未产出结果。',
        'failed', 'manifest.json status=failed，无完成单元。'))
    documents.append(empty_document(
        'results_phase2_confirmation_v2', 'phase2 确认线 v2（模型复用审计）', 'phase1~4 确认线',
        'phase2 模型复用与确认的审计产物（proposed_manifest、reuse 审计），无独立评估指标行。',
        'empty', '仅复用/审计清单，无 success_rate。'))
    documents.append(empty_document(
        'results_phase2_diagnosis_20260908', 'phase2 诊断线（工程迭代与 tensorboard 诊断）', 'phase1~4 确认线',
        'phase2 运行时/工程诊断（confirmation planner、factory tests、tensorboard csv），非正式评估结果。',
        'empty', '诊断与测试产物，无标准指标。'))
    documents.append(empty_document(
        'results_phase2_failed_parameters_by_scenario_v1', 'phase2 失败参数曲线线（仅图，无表）', 'phase1~4 确认线',
        '按场景绘制失败参数组合的训练曲线（pdf/png/svg），未导出结构化指标表。',
        'empty', '仅曲线图，无结构化指标文件。'))
    documents.append(empty_document(
        'results_phase2_reuse_v3', 'phase2 复用分析线 v3（筛选决策）', 'phase1~4 确认线',
        'phase2 模型复用分析与筛选决策（completion/evidence_sources/screen_decision）。',
        'empty', '筛选决策产物，无评估指标。'))
    documents.append(empty_document(
        'results_phase2_runtime_v1', 'phase2 运行时线 v1（preflight）', 'phase1~4 确认线',
        'phase2 运行时 v1 仅 preflight_v1.json（预检），被 runtime_v2 取代。',
        'empty', '仅 preflight。'))
    documents.append(empty_document(
        'results_phase2_training_curves_v2_episode20_ema999', 'phase2 训练曲线线 v2（episode20/ema999）', 'phase1~4 确认线',
        '训练曲线值（curve_values.json，8MB）与 pdf/png/svg 曲线图；曲线数值未标准化为逐单元指标行。',
        'empty', '曲线数据，未提取为标准指标行。'))
    documents.append(empty_document(
        'results_method_candidate_assessment_20260912_v1', '方法候选评估线（20260912）', 'phase1~4 确认线',
        '方法候选证据审计（evidence_audit.json + REPORT.md），为人工报告而非逐运行结果。',
        'empty', '证据审计与报告，无逐运行指标。'))
    documents.append(empty_document(
        'results_promotion_automation', '晋升自动化记录线', '工程自动化',
        '各版本晋升自动化的注册/挂起/恢复状态清单与 detached 日志，非科学结果。',
        'empty', '自动化状态记录。'))

    for doc in documents:
        target = RAW / f"{doc['directory']}.json"
        target.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')

    total_rows = sum(len(d['rows']) for d in documents)
    print(json.dumps({'written': len(documents), 'total_rows': total_rows,
                      'directories': [d['directory'] for d in documents]},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
