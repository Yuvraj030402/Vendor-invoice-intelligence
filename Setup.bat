@echo off
cd /d "%~dp0"
python bootstrap.py --setup-only
if errorlevel 1 pause
