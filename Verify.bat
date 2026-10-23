@echo off
cd /d "%~dp0"
python bootstrap.py --verify
if errorlevel 1 pause
