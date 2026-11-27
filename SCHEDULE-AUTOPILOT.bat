@echo off
rem Creates a Windows Task Scheduler job that runs AUTOPILOT.bat at 09:00 every weekday.
rem Remove it any time with:  schtasks /Delete /TN "Quant Paper Autopilot" /F
cd /d %~dp0
schtasks /Create /TN "Quant Paper Autopilot" /TR "\"%~dp0AUTOPILOT.bat\"" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 09:00 /F
if errorlevel 1 (
  echo.
  echo Could not create the task. Try right-clicking this file and choosing "Run as administrator".
  pause
  exit /b 1
)
echo.
echo Done. Every weekday at 09:00 your computer will run the paper-trading autopilot
echo (live session until 15:30, then fetch, retrain and the evening agent cycle).
echo Keep the computer ON and AWAKE during market hours, and logged in.
echo Log file: %~dp0data\autopilot.log
echo Check the task:  schtasks /Query /TN "Quant Paper Autopilot"
echo Remove the task: schtasks /Delete /TN "Quant Paper Autopilot" /F
pause
