"""Merge the per-directory extraction JSONs into one categorized Excel workbook.

Every sheet is derived from ``raw/*.json``; nothing is imputed.  A row whose
numbers could not be traced to a real artifact carries nulls and a note, never a
guess.
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / 'raw'
WORKBOOK = HERE / '实验数据汇总_全版本.xlsx'

ROW_COLUMNS = [
    ('batch', '批次目录'), ('experiment_stage', '实验阶段'), ('phase', '阶段/版本线'),
    ('variant', '方法/变体'), ('candidate', '参数候选'), ('scenario', '场景'),
    ('training_seed', '训练种子'), ('model_kind', '模型种类'), ('checkpoint_step', '检查点步数'),
    ('episodes', '评估回合数'), ('eval_seed_start', '评估种子起点'),
    ('decoder', '解码器'), ('success_rate', '成功率'), ('collision_rate', '碰撞率'),
    ('timeout_rate', '超时率'), ('mean_return', '平均回报'), ('status', '状态'),
    ('evidence_kind', '证据类型'), ('artifact', '源文件'), ('notes', '备注'),
]
METRIC_COLUMNS = ('成功率', '碰撞率', '超时率', '平均回报')

SCENARIO_ALIASES = {
    'full_balanced': 'full_balanced', 'left_turn': 'left_turn', 'cross': 'cross',
    'roundabout': 'roundabout', 'roundabout_easy': 'roundabout_easy',
    'roundabout_medium': 'roundabout_medium', 'carla': 'carla', 'ram': 'ramp',
    'x': 'x', 'highway': 'highway',
}

# 高层「实验阶段」分类：覆盖全部批次目录（含无指标的过程性目录）。
EXPERIMENT_STAGE_MAP = {
    # 系统矩阵
    'results_systematic_matrix': '系统矩阵',
    # 独立 v2
    'results_iv2_5m6s100e_v1': '独立v2',
    'results_iv2_eval_100s100e_v1': '独立v2',
    'results_iv2_eval_3m_100s100e_v1': '独立v2',
    'results_visual_eval_three_methods_20ep_v1': '独立v2',
    # 高密度对比
    'results_hd_ss100_v2': '高密度对比',
    'results_hd_ss100_s20_v3': '高密度对比',
    'results_high_density_same_scene_v1': '高密度对比',
    # topo v1~v3
    'results_topo_scene': 'topo v1~v3',
    'results_topo_v2': 'topo v1~v3',
    'results_topo_v2_fast': 'topo v1~v3',
    'results_topo_v3_dev': 'topo v1~v3',
    # topo v4 版本线（dev / promotion 全部）
    'results_topo_v4_dev': 'topo v4 版本线',
    'results_topo_v4_2_dev': 'topo v4 版本线',
    'results_topo_v4_2_r1_dev': 'topo v4 版本线',
    'results_topo_v4_3_dev': 'topo v4 版本线',
    'results_topo_v4_4_dev': 'topo v4 版本线',
    'results_topo_v4_5_dev': 'topo v4 版本线',
    'results_topo_v4_6_dev': 'topo v4 版本线',
    'results_topo_v4_7_dev': 'topo v4 版本线',
    'results_topo_v4_7_1_dev': 'topo v4 版本线',
    'results_topo_v4_7_2_dev': 'topo v4 版本线',
    'results_topo_v4_8_dev': 'topo v4 版本线',
    'results_topo_v4_8_1_dev': 'topo v4 版本线',
    'results_topo_v4_8_2_dev': 'topo v4 版本线',
    'results_topo_v4_8_3_dev': 'topo v4 版本线',
    'results_topo_v4_8_promotion': 'topo v4 版本线',
    'results_topo_v4_9_dev': 'topo v4 版本线',
    'results_topo_v4_9_1_dev': 'topo v4 版本线',
    'results_topo_v4_9_2_dev': 'topo v4 版本线',
    'results_topo_v4_9_2_promotion': 'topo v4 版本线',
    'results_topo_v4_10': 'topo v4 版本线',
    'results_topo_v4_10_dev': 'topo v4 版本线',
    'results_topo_v4_11_dev': 'topo v4 版本线',
    'results_topo_v4_11_promotion': 'topo v4 版本线',
    'results_topo_v4_12_dev': 'topo v4 版本线',
    'results_topo_v4_13_dev': 'topo v4 版本线',
    'results_topo_v4_13_promotion': 'topo v4 版本线',
    # phase1~4 正式确认线
    'results_phase1_checkpoint_diagnostics_v1': 'phase1~4 确认线',
    'results_phase2_runtime_v1': 'phase1~4 确认线',
    'results_phase2_runtime_v2': 'phase1~4 确认线',
    'results_phase2_systematic_report_v1': 'phase1~4 确认线',
    'results_phase2_confirmation_v2': 'phase1~4 确认线',
    'results_phase2_diagnosis_20260908': 'phase1~4 确认线',
    'results_phase2_failed_parameters_by_scenario_v1': 'phase1~4 确认线',
    'results_phase2_reuse_v3': 'phase1~4 确认线',
    'results_phase2_training_curves_v2_episode20_ema999': 'phase1~4 确认线',
    'results_method_candidate_assessment_20260912_v1': 'phase1~4 确认线',
    'r3m1': 'phase1~4 确认线',
    'r4m1': 'phase1~4 确认线',
    # 工程自动化
    'results_promotion_automation': '工程自动化',
}


def stage_of(document):
    return (document.get('experiment_stage')
            or EXPERIMENT_STAGE_MAP.get(document['directory'], '未分类'))


def load_documents():
    documents = []
    for path in sorted(RAW.glob('*.json')):
        try:
            document = json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:  # a broken extraction must be visible, not skipped
            documents.append({'directory': path.stem, 'phase': None, 'purpose': None,
                              'status': 'UNREADABLE', 'rows': [], 'missing': [],
                              'notes': f'读取失败：{exc!r}'})
            continue
        document.setdefault('directory', path.stem)
        document.setdefault('rows', [])
        document.setdefault('missing', [])
        documents.append(document)
    return documents


def flatten(document):
    rows = []
    stage = stage_of(document)
    for raw in document.get('rows') or []:
        if not isinstance(raw, dict):
            continue
        row = {chinese: raw.get(english) for english, chinese in ROW_COLUMNS}
        row['批次目录'] = row['批次目录'] or document['directory']
        row['实验阶段'] = row['实验阶段'] or stage
        row['阶段/版本线'] = row['阶段/版本线'] or document.get('phase')
        row['场景'] = SCENARIO_ALIASES.get(str(row['场景']), row['场景'])
        keys = raw.get('source_keys')
        row['指标来源键'] = json.dumps(keys, ensure_ascii=False) if keys else None
        rows.append(row)
    return rows


def main():
    documents = load_documents()
    frames = []
    for document in documents:
        rows = flatten(document)
        if rows:
            frames.append(pd.DataFrame(rows))
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[c for _, c in ROW_COLUMNS])
    table = table.drop_duplicates()

    # 冒烟证据（evidence_kind == 'smoke'）从主结果中排除，单独成表；符合「冒烟实验除外」。
    is_smoke = table['证据类型'] == 'smoke'
    main_table = table[~is_smoke].copy()
    smoke_table = table[is_smoke].copy()

    index = pd.DataFrame([{
        '批次目录': d['directory'], '实验阶段': stage_of(d), '阶段/版本线': d.get('phase'),
        '批次用途': d.get('purpose'), '批次状态': d.get('status'), '数据行数': len(flatten(d)),
        '缺失单元数': len(d.get('missing') or []),
        '协议文件': ', '.join(d.get('protocol_files') or []) or None,
        '备注': d.get('notes'),
    } for d in documents]).sort_values(['实验阶段', '数据行数'], ascending=[True, False])

    # Aggregate only over rows that actually carry the metric; group sizes stay visible.
    stage_table = (main_table.dropna(subset=['成功率'])
                   .groupby(['实验阶段', '阶段/版本线', '方法/变体'], dropna=False)
                   .agg(单元数=('成功率', 'size'), 场景数=('场景', 'nunique'),
                        训练种子数=('训练种子', 'nunique'),
                        成功率均值=('成功率', 'mean'), 成功率标准差=('成功率', 'std'),
                        碰撞率均值=('碰撞率', 'mean'), 超时率均值=('超时率', 'mean'),
                        平均回报均值=('平均回报', 'mean'))
                   .reset_index().sort_values(['实验阶段', '阶段/版本线', '方法/变体']))

    grouped = (main_table.dropna(subset=['成功率'])
               .groupby(['实验阶段', '阶段/版本线', '方法/变体', '参数候选', '模型种类'], dropna=False)
               .agg(单元数=('成功率', 'size'), 场景数=('场景', 'nunique'),
                    训练种子数=('训练种子', 'nunique'),
                    成功率均值=('成功率', 'mean'), 成功率标准差=('成功率', 'std'),
                    碰撞率均值=('碰撞率', 'mean'), 超时率均值=('超时率', 'mean'),
                    平均回报均值=('平均回报', 'mean'))
               .reset_index().sort_values(['实验阶段', '阶段/版本线', '方法/变体']))

    scenario_table = (main_table.dropna(subset=['成功率'])
                      .groupby(['实验阶段', '阶段/版本线', '方法/变体', '场景', '模型种类'], dropna=False)
                      .agg(单元数=('成功率', 'size'), 成功率均值=('成功率', 'mean'),
                           成功率标准差=('成功率', 'std'), 碰撞率均值=('碰撞率', 'mean'),
                           超时率均值=('超时率', 'mean'), 平均回报均值=('平均回报', 'mean'))
                      .reset_index().sort_values(['实验阶段', '阶段/版本线', '方法/变体', '场景']))

    missing_rows = []
    for document in documents:
        for item in document.get('missing') or []:
            if isinstance(item, dict):
                missing_rows.append({'批次目录': document['directory'],
                                     '缺失单元': item.get('unit'), '原因': item.get('reason')})
            else:
                missing_rows.append({'批次目录': document['directory'], '缺失单元': str(item),
                                     '原因': None})
    missing_table = pd.DataFrame(missing_rows)

    audit = pd.DataFrame([{
        '批次目录': d['directory'], '实验阶段': stage_of(d), '抽取状态': d.get('status'),
        '数据行数': len(flatten(d)), '备注': d.get('notes'),
    } for d in documents])

    overview = pd.DataFrame([
        {'项目': '数据来源目录数', '值': len(documents)},
        {'项目': '主结果数据行数（已排除冒烟）', '值': len(main_table)},
        {'项目': '含成功率的行数', '值': int(main_table['成功率'].notna().sum())},
        {'项目': '冒烟证据行数（已单独分离）', '值': len(smoke_table)},
        {'项目': '涉及实验阶段数', '值': main_table['实验阶段'].nunique(dropna=True)},
        {'项目': '涉及阶段/版本线数', '值': main_table['阶段/版本线'].nunique(dropna=True)},
        {'项目': '涉及方法/变体数', '值': main_table['方法/变体'].nunique(dropna=True)},
        {'项目': '涉及场景数', '值': main_table['场景'].nunique(dropna=True)},
        {'项目': '涉及训练种子数', '值': main_table['训练种子'].nunique(dropna=True)},
        {'项目': '实验阶段分布', '值': json.dumps(main_table['实验阶段'].value_counts(dropna=False).to_dict(), ensure_ascii=False)},
        {'项目': '状态分布', '值': json.dumps(main_table['状态'].value_counts(dropna=False).to_dict(), ensure_ascii=False)},
        {'项目': '证据类型分布', '值': json.dumps(main_table['证据类型'].value_counts(dropna=False).to_dict(), ensure_ascii=False)},
        {'项目': '缺失单元条目数', '值': len(missing_table)},
        {'项目': '说明', '值': '全部数值取自磁盘上真实文件；取不到一律留空，绝不插补。冒烟实验（smoke/preflight）已从主结果排除，单独放入「冒烟证据(已排除)」表。'},
    ])

    with pd.ExcelWriter(WORKBOOK, engine='openpyxl') as writer:
        overview.to_excel(writer, sheet_name='总览', index=False)
        main_table.to_excel(writer, sheet_name='全部结果明细', index=False)
        stage_table.to_excel(writer, sheet_name='按实验阶段汇总', index=False)
        grouped.to_excel(writer, sheet_name='按方法汇总', index=False)
        scenario_table.to_excel(writer, sheet_name='按场景汇总', index=False)
        smoke_table.to_excel(writer, sheet_name='冒烟证据(已排除)', index=False)
        index.to_excel(writer, sheet_name='批次索引', index=False)
        missing_table.to_excel(writer, sheet_name='缺失与中止', index=False)
        audit.to_excel(writer, sheet_name='逐目录抽取核查', index=False)

        workbook = writer.book
        from openpyxl.styles import Alignment, Font
        from openpyxl.utils import get_column_letter
        for sheet in workbook.worksheets:
            for cell in sheet[1]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(vertical='center', wrap_text=True)
            sheet.freeze_panes = 'A2'
            for column in range(1, sheet.max_column + 1):
                width = max((len(str(sheet.cell(row=r, column=column).value or ''))
                             for r in range(1, min(sheet.max_row, 400) + 1)), default=8)
                sheet.column_dimensions[get_column_letter(column)].width = min(48, max(10, width + 2))
        for sheet_name in ('全部结果明细', '按实验阶段汇总', '按方法汇总', '按场景汇总'):
            sheet = workbook[sheet_name]
            header = {cell.value: cell.column for cell in sheet[1]}
            for name in ('成功率', '碰撞率', '超时率', '成功率均值', '碰撞率均值', '超时率均值',
                         '成功率标准差'):
                if name in header:
                    for row in range(2, sheet.max_row + 1):
                        sheet.cell(row=row, column=header[name]).number_format = '0.0000'

    return {'workbook': str(WORKBOOK), 'directories': len(documents), 'rows': len(main_table),
            'rows_with_success_rate': int(main_table['成功率'].notna().sum()),
            'smoke_rows': len(smoke_table), 'missing_entries': len(missing_table)}


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2))
