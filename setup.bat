@echo off
REM ----------------------------------------------------------------------
REM setup.bat  -  one-shot first-time setup for a fresh Windows install.
REM
REM Creates an isolated Python virtualenv at .venv\ and installs the
REM backend deps inside it, then installs the frontend deps. Once
REM finished, dev.bat / start.bat auto-detect the venv and use it.
REM
REM Why a venv: installing into the system + user site-packages dirs
REM mixes locations (some packages land in C:\Python3xx\Lib\, others
REM in %APPDATA%\Python\Python3xx\), and Windows file ACLs / antivirus
REM scans / OneDrive sync can lock individual files mid-install. The
REM result is a "PermissionError: [Errno 13] Permission denied" the
REM next time `python -m uvicorn` tries to import one of those files.
REM A venv keeps every dep inside the project folder where only this
REM project touches it.
REM
REM Re-runnable: pip is idempotent on already-satisfied deps, so it's
REM also the right thing to run after pulling new commits if you want
REM to make sure backend/requirements.txt is in sync.
REM
REM Requires: Python 3.12+, Node 20+. Ollama is auto-started by the
REM backend at first launch if it's installed; otherwise install from
REM https://ollama.com/download.
REM ----------------------------------------------------------------------

setlocal
set ROOT=%~dp0
cd /d %ROOT%

echo.
echo  ====================================================
echo   Gigachat first-time setup
echo  ====================================================
echo.

REM ----- Python venv ----------------------------------------------------
if not exist "%ROOT%.venv\Scripts\python.exe" (
  echo  [1/3] Creating Python virtualenv at .venv\ ...
  python -m venv .venv
  if errorlevel 1 (
    echo.
    echo  [!] Failed to create .venv. Make sure Python 3.12+ is on PATH:
    echo        python --version
    exit /b 1
  )
) else (
  echo  [1/3] Reusing existing .venv\ ...
)

REM ----- Backend deps ---------------------------------------------------
echo  [2/3] Installing backend dependencies into .venv\ ...
call "%ROOT%.venv\Scripts\python.exe" -m pip install --upgrade pip
call "%ROOT%.venv\Scripts\python.exe" -m pip install -r backend\requirements.txt
if errorlevel 1 (
  echo.
  echo  [!] Backend dep install failed. Common causes:
  echo        - Antivirus scanning the venv mid-write. Add %ROOT%.venv\ to
  echo          your AV's exclusion list and retry.
  echo        - Network blocked. Try `python -m pip install --upgrade pip`
  echo          first to confirm PyPI is reachable.
  exit /b 1
)

REM ----- Frontend deps --------------------------------------------------
echo  [3/3] Installing frontend dependencies via npm ...
cd /d %ROOT%frontend
call npm install
if errorlevel 1 (
  echo.
  echo  [!] npm install failed. Make sure Node 20+ is on PATH:
  echo        node --version
  exit /b 1
)
cd /d %ROOT%

echo.
echo  ====================================================
echo   Setup complete.
echo.
echo   Run `dev.bat` for hot-reload development at
echo     http://localhost:5173
echo.
echo   Run `start.bat` for the production single-port build at
echo     http://localhost:8000
echo  ====================================================
echo.

endlocal
