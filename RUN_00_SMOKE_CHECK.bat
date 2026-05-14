@echo off
setlocal
cd /d %~dp0
python scripts\smoke_check_literace.py
if errorlevel 1 goto failed
echo.
echo [OK] LiteRaceSegNet imports and forward pass work without a dataset.
pause
exit /b 0
:failed
echo.
echo [FAILED] Install requirements first: 00_INSTALL_REQUIREMENTS.bat
pause
exit /b 1
