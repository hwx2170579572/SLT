"""补评估：4 个 route 方法续训到 100k 的训练已完成（final_model.zip 已保存），
但评估阶段因 MAX_PATH（``_evaluate_saved`` 的 overlay 目录过长）报 WinError 3 失败。

本脚本只补跑评估（CPU）并补写 continuation / status 产物，不重新训练。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_intersection_yield_v2_d1 as d1

DEPART = 4.0
METHODS = [
    "sac_mlp_d1_st_rt_late",
    "sac_mlp_d1_st_rt_gate",
    "sac_mlp_d1_st_rt_ego",
    "sac_mlp_d1_st_rt_edge",
]


def main() -> int:
    d1._apply_patch(DEPART)
    d1.ensure_sorted_scenario()

    for gname in METHODS:
        family, _mod, _method = d1.DISPATCH[gname]
        run_dir = d1._run_dir(gname, DEPART)
        cont_dir = run_dir / "c100000"
        final = cont_dir / "final_model.zip"
        if not final.is_file():
            print(f"[skip] {gname}: no final_model", flush=True)
            continue

        args = json.loads((cont_dir / "arguments.json").read_text(encoding="utf-8"))
        result = d1.evaluate(gname, cont_dir, final, smoke=False)
        summary = result["summary"]

        manifest = dict(
            method=gname,
            family=family,
            scenario=d1.NEW_SCENARIO,
            depart_scale=DEPART,
            run_dir=str(run_dir),
            base_model=args["base_model"],
            base_raw_steps=args["base_raw_steps"],
            extra_raw_steps=args["extra_raw_steps"],
            global_raw_steps=args["global_raw_steps"],
            continuation_dir=str(cont_dir),
            final_model=str(final),
            evaluation=summary,
            smoke=False,
            note=(
                "续训：加载 base_model 权重/优化器/计数器；replay buffer 从空重新累积；"
                "raw_learning_starts 设 500 先填满 buffer；checkpoint 为局部步数，"
                "保存前把 _raw_steps_seen 写回全局步数。（评估阶段补跑，训练已完成）"
            ),
        )
        d1.base._write_json_atomic(
            run_dir / f"continuation_{args['global_raw_steps']}.json", manifest
        )
        d1.base._write_json_atomic(
            cont_dir / "status.json", dict(status="trained", smoke=False)
        )
        print(
            json.dumps(dict(method=gname, summary=summary), ensure_ascii=False, indent=2),
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
