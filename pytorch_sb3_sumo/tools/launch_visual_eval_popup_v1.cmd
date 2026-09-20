@echo off
setlocal
cd /d "%~dp0.."

rem New-file launcher for the 20-episode live SUMO-GUI evaluation.
rem Double-click this file from the Windows desktop/File Explorer so the
rem SUMO-GUI child is created by the interactive user desktop.
start "SUMO live evaluation" /normal "D:\Programs\Anaconda\envs\llm_pipeline\pythonw.exe" "tools\evaluate_visual_three_methods_six_scenes_20ep_v1.py" --live --speed 4 --device cuda

endlocal
