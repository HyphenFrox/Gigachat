@echo off
REM ----------------------------------------------------------------------
REM build.bat  -  build the frontend for production.
REM
REM Outputs to frontend/dist/. After this, `start.bat` serves everything
REM from the FastAPI backend at http://localhost:8000.
REM ----------------------------------------------------------------------

setlocal
set ROOT=%~dp0
cd /d %ROOT%frontend
call npm run build
endlocal
