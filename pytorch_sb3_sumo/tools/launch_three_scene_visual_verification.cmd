@echo off
setlocal
cd /d "%~dp0.."

rem Desktop launcher for the live SUMO-GUI visual verification of the nine
rem three-scene methods' final models.  Double-click this file from the Windows
rem desktop/File Explorer so the SUMO-GUI child is created by the interactive
rem user desktop (the script also forces SUMO-GUI visibility on Windows).
rem
rem Opens an interactive window where you tick the methods and scenarios to
rem run and press start.  Results land under
rem results_three_scene_v1\visual_verification\live\<timestamp>.
start "Three-scene visual verification" /normal "D:\Programs\Anaconda\envs\llm_pipeline\pythonw.exe" "tools\three_scene_visual_verification.py" --gui --device cuda

endlocal
