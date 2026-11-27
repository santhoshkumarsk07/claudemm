@echo off
rem Runs one full paper-trading day: live session 09:15-15:30, then fetch, retrain and agents.
rem Scheduled by SCHEDULE-AUTOPILOT.bat. Output goes to data\autopilot.log. Paper money only.
cd /d %~dp0
if not exist data mkdir data
set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8
echo ===== %date% %time% using %PY% ===== >> data\autopilot.log
%PY% -m quant autopilot >> data\autopilot.log 2>&1
