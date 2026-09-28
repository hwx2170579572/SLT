@echo off
setlocal
cd /d "%~dp0.."

rem Desktop launcher for the live SUMO-GUI driving replay of every saved final
rem model under fast-developer, across all scenarios and all methods.
rem Scenarios (intersection, the intersection_sorted depart-sorted copy, and the
rem depart-scale groups) and methods are registered in visual_eval_driving.py
rem (METHODS / SCENARIOS tables); this launcher only opens the GUI, which
rem auto-discovers every method that has a saved checkpoint (final_model.zip or
rem best_training_success_model.zip).
rem Recent additions (intersection_sorted + depart dimension):
rem   - sac_mlp_depart{2p5,3p0,4p0}__intersection_sorted (depart-scale density sweep)
rem   - mst_slt__intersection_sorted_depart4p0, sac_mlp_d1_st_rt__intersection_sorted_depart4p0,
rem     sac_mlp_d1_st__intersection_sorted_depart4p0 (orchestrator train_intersection_yield_v2_d1.py)
rem Double-click this file from the Windows desktop / File Explorer to open an
rem interactive window where you tick the methods to run, set episodes / speed /
rem device, then press start.
rem Results land under fast-developer\_visual_eval_driving\live\<timestamp>.
rem NOTE: this file must stay ASCII-only; cmd.exe reads .cmd files as GBK and
rem mangles UTF-8 Chinese text.
start "fast-developer live driving" /normal "D:\Programs\Anaconda\envs\llm_pipeline\pythonw.exe" "fast-developer\visual_eval_driving.py" --gui --device cuda

endlocal
