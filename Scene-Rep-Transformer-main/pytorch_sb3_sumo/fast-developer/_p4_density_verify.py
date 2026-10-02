"""P4 密度验证：生成 depart×scale 的低密度 traffic 变体，测「不干预」让行成功率。

目的：验证「密度是 intersection 让行的根本瓶颈」。若低密度下「不干预」成功率显著
上升（例如 scale=4 时 >80%），则证明课程学习（先低密度学会让行、再加密回原密度）是
正解；若低密度下仍低，则瓶颈在 unregulated junction 的冲突机制而非密度。

做法：读原始 traffic_*.rou.xml，把每个 vehicle 的 depart 时间 ×scale，写到
fast-developer/_p4_lowdensity/，再 monkey-patch 环境的 traffic 路径指向低密度文件，
跑「不干预」（setSpeedMode 31，不 setSpeed）。

用法::

    python fast-developer/_p4_density_verify.py 30 2.0   # 30 集, 密度减半
    python fast-developer/_p4_density_verify.py 30 4.0   # 30 集, 密度 1/4
"""
import sys
import xml.etree.ElementTree as ET
from math import hypot
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import train_intersection_hold35k_mst_fixed as base

SOURCE_TRAFFIC = _PROJECT_ROOT / "envs" / "sumo" / "original_scenarios_v1" / "intersection" / "traffic"


def _make_low_density(scale: float, out_dir: Path) -> tuple[Path, ...]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for src in sorted(SOURCE_TRAFFIC.glob("traffic_*.rou.xml")):
        tree = ET.parse(src)
        root = tree.getroot()
        n_scaled = 0
        for veh in root.findall("vehicle"):
            depart = float(veh.attrib.get("depart", "0"))
            veh.attrib["depart"] = f"{depart * scale:.3f}"
            n_scaled += 1
        dst = out_dir / src.name
        tree.write(dst, encoding="UTF-8", xml_declaration=True)
        paths.append(dst)
    print(f"生成 {len(paths)} 个低密度 traffic (scale={scale}) -> {out_dir}", flush=True)
    return tuple(paths)


def run_nointervention(n_ep: int, traffic_paths: tuple[Path, ...], overlay_root: Path) -> dict:
    env = base.make_env_factory("base", overlay_root)(base._environment_namespace(), evaluation=False)
    env._partitioned_traffic_paths = lambda spec: traffic_paths
    succ = coll = timeout = 0
    for ep in range(n_ep):
        env._traffic_episode_index = ep
        env._traffic_roll = None
        obs, info = env.reset(seed=base.SEED_START["mst_slt"] + ep)
        conn = env._connection
        ego_id = env.specification.ego_id
        conn.vehicle.setSpeedMode(ego_id, 31)
        done = False
        raw = 0
        while not done and raw < env.max_episode_steps:
            conn.simulationStep()
            env._after_simulation_step()
            env._raw_steps += 1
            env._record_histories()
            success, collision, off_route, max_time = env._events_after_step()
            raw += 1
            if success or collision or off_route or max_time:
                done = True
        succ += int(success)
        coll += int(collision)
        timeout += int(max_time)
    env.close()
    return dict(success=succ, collision=coll, timeout=timeout, n=n_ep)


def main():
    import shutil

    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    scale_token = str(scale).replace(".", "p")
    out_dir = _THIS_DIR / f"_p4_lowdensity_s{scale_token}"
    paths = _make_low_density(scale, out_dir)
    overlay_root = _THIS_DIR / f"_p4_verify_overlay_s{scale_token}"
    shutil.rmtree(overlay_root, ignore_errors=True)
    result = run_nointervention(n_ep, paths, overlay_root)
    print(f"scale={scale} 不干预: success={result['success']}/{n_ep} "
          f"collision={result['collision']} timeout={result['timeout']}", flush=True)


if __name__ == "__main__":
    main()
