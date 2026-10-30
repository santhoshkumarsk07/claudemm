@echo off
cd /d %~dp0
set PY=python
where python >nul 2>nul || set PY=py -3
%PY% -c "import sys; assert sys.version_info >= (3, 11), 'Python 3.11+ is required'" || (
  echo.
  echo Python 3.11 or newer was not found. Install it from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during install, then run this again.
  pause
  exit /b 1
)
%PY% -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip -q
.venv\Scripts\python -m pip install --require-hashes -r requirements.lock -q
if not exist .env copy .env.example .env >nul
echo.
echo Setup done. Double-click START-TRADING.bat to open the Arena.
