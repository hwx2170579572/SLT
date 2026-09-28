@echo off
setlocal
cd /d "%~dp0.."

rem Desktop launcher for the live SUMO-GUI driving replay of every saved final
rem model under fast-developer, across all scenarios and all methods.
rem Scenarios (intersection, the intersection_sorted depart-sorted copy, and the
rem sac_mlp depart-scale groups) and methods are registered in
rem visual_eval_driving.py (METHODS / SCENARIOS tables); this launcher only opens
rem the GUI, which auto-discovers every method that has a saved checkpoint
rem (final_model.zip or best_training_success_model.zip).
rem Double-click this file from the Windows desktop / File Explorer to open an
rem interactive window where you tick the methods to run, set episodes / speed /
rem device, then press start.
rem Results land under fast-developer\_visual_eval_driving\live\<timestamp>.
rem NOTE: this file must stay ASCII-only; cmd.exe reads .cmd files as GBK and
rem mangles UTF-8 Chinese text.
start "fast-developer live driving" /normal "D:\Programs\Anaconda\envs\llm_pipeline\pythonw.exe" "fast-developer\visual_eval_driving.py" --gui --device cuda

endlocal
