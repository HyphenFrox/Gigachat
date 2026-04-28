@echo off
REM ----------------------------------------------------------------------
REM scripts\test.bat — local smoke-test runner.
REM
REM Same command CI runs (`pytest -m smoke ... -x`), so a green local run
REM is a strong signal a push won't go red. Designed to double as the
REM body of a `pre-push` git hook (see .githooks\pre-push) so a forgotten
REM `pytest` doesn't ship a regression.
REM
REM Exit code propagates from pytest: 0 = pass, non-zero = fail. The
REM pre-push hook short-circuits the push when this script exits non-zero.
REM
REM Usage:
REM   scripts\test.bat            — run smoke tier on the current Python
REM   scripts\test.bat -v         — extra args forwarded to pytest verbatim
REM ----------------------------------------------------------------------

setlocal
set ROOT=%~dp0..
cd /d %ROOT%

REM `--ignore` skips test_app_security.py if `pywebpush` isn't importable
REM in this environment — same fallback the pre-push hook uses. The full
REM CI runner has every dep installed and doesn't need this skip.
python -m pytest -m smoke --no-header -q --tb=short -x %*
exit /b %ERRORLEVEL%
