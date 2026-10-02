#!/bin/bash
# D1 表征消融 8 臂 50k 满预算训练 launcher。
# 单 GPU（16GB）+ ~11GB 可用 RAM，故分 2 批，每批 4 臂并行（连续头一批、混合头一批）。
set -u
cd "$(dirname "$0")"

PY="D:/Programs/Anaconda/envs/llm_pipeline/python.exe"
SCRIPT="train_intersection_yield_v2_d1.py"
STEPS=50000

echo "[$(date +%H:%M:%S)] batch 1: continuous-head 4 arms"
for m in sac_mlp_d1_st sac_mlp_d1_st_rt sac_mlp_d1_st_rt_topo sac_mlp_d1_full; do
  "$PY" "$SCRIPT" --method "$m" --max-steps "$STEPS" > "train_${m}.log" 2>&1 &
done
wait

echo "[$(date +%H:%M:%S)] batch 2: hybrid-head 4 arms"
for m in hsac_mlp_base_d1_st hsac_mlp_base_d1_st_rt hsac_mlp_base_d1_st_rt_topo hsac_mlp_base_d1_full; do
  "$PY" "$SCRIPT" --method "$m" --max-steps "$STEPS" > "train_${m}.log" 2>&1 &
done
wait

echo "[$(date +%H:%M:%S)] ALL D1 TRAINING DONE"
