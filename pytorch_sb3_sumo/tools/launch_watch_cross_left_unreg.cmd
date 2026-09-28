@echo off
setlocal
cd /d "%~dp0.."

rem Desktop launcher for the live SUMO-GUI preview of the new cross_left_unreg
rem scenario (random policy, no checkpoint).  Double-click to watch the
rem uncontrolled 4-way intersection with 0.2 veh/s per approach before or while
rem it is being trained.  Edit the flags below to change episodes / speed.
start "Watch cross_left_unreg" /normal "D:\Programs\Anaconda\envs\llm_pipeline\pythonw.exe" "tools\watch_cross_left_unreg.py" --scenario cross_left_unreg --method hold35k --episodes 3 --speed 1.0

endlocal
