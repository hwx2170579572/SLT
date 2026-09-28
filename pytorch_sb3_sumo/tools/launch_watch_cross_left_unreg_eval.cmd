@echo off
setlocal
cd /d "%~dp0.."

rem Desktop launcher for the live SUMO-GUI evaluation playback of the trained
rem cross_left_unreg policies (hold35k / mst_slt).  Double-click to watch the
rem ego vehicle drive the uncontrolled 4-way junction with the trained policy.
rem
rem Edit the flags below to change method / checkpoint / episodes / speed.
rem Use --final to load final_model.zip, or --raw-step N for a specific step.

start "Watch cross_left_unreg eval (hold35k)" /normal "D:\Programs\Anaconda\envs\llm_pipeline\python.exe" "tools\watch_cross_left_unreg_eval.py" --method hold35k --raw-step 10000 --episodes 3 --speed 1.0

endlocal
