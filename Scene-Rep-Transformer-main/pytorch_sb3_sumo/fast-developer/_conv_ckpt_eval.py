import sys, json, re
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
FAST_DEV = Path(__file__).resolve().parent
sys.path.insert(0, str(FAST_DEV))
import numpy as np, torch
torch.set_num_threads(1)
import train_intersection_yield_v2 as yv2

yv2.SCENARIO = "intersection_sorted"
yv2.SOURCE_TRAFFIC = PROJECT_ROOT / "envs" / "sumo" / "original_scenarios_v1" / "intersection_sorted" / "traffic"
yv2.DEPART_SCALE = 4.0

from algos.sb3_torch.sac import SceneRepresentationSAC
from algos.sb3_torch.evaluation import evaluate_model_detailed

targets = [
    ("mst_slt",            "mst_slt__intersection_sorted_depart4p0"),
    ("sac_mlp_d1_st",      "sac_mlp_d1_st__intersection_sorted_depart4p0"),
    ("sac_mlp_d1_st_attn", "sac_mlp_d1_st_attn__intersection_sorted_depart4p0"),
    ("sac_mlp_d1_st_rt",   "sac_mlp_d1_st_rt__intersection_sorted_depart4p0"),
    ("sac_mlp_baseline",   "sac_mlp_depart4p0__intersection_sorted"),
]
STEPS = [20000, 35000]
EP = 50
SEED_START = 10_000

def step_of(p):
    m = re.search(r"raw_(\d+)_steps", p.name)
    return int(m.group(1)) if m else 10**9

def find_ckpt(run_dir, st):
    ckpts = list(run_dir.glob("checkpoints/*_steps.zip"))
    if not ckpts:
        raise RuntimeError(f"no checkpoints under {run_dir}")
    return min(ckpts, key=lambda p: abs(step_of(p) - st))

results = {}
for label, d in targets:
    run_dir = FAST_DEV / d
    env = yv2.make_env_factory("base", run_dir / "overlays" / "eval")(
        yv2._environment_namespace(), evaluation=True
    )
    for st in STEPS:
        ckpt = find_ckpt(run_dir, st)
        model = SceneRepresentationSAC.load(str(ckpt), env=env, device="cuda", buffer_size=32)
        succ = coll = 0
        for i in range(EP):
            np.random.seed(SEED_START + i + 600_000)
            torch.manual_seed(SEED_START + i + 600_000)
            env.unwrapped._traffic_episode_index = i
            env.unwrapped._traffic_roll = None
            det = evaluate_model_detailed(
                model, env, episodes=1, seed=SEED_START + i, deterministic=True,
                sumo_step_seconds=0.1, policy_action_hold=1,
            )
            rec = det.episode_records[0].to_dict()
            succ += int(rec["success"]); coll += int(rec["collision"])
        key = f"{label}@{st}"
        results[key] = dict(success=succ / EP, collision=coll / EP, n=EP, ckpt=ckpt.name)
        print(f"{key}: success={succ/EP:.3f} collision={coll/EP:.3f} ({ckpt.name})", flush=True)
    env.close()

Path("_conv_checkpoints.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
print("DONE", flush=True)
