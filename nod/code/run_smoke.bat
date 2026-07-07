@echo off
REM Smoke test: ~minutes on RTX 4080
"C:\Users\luisl\anaconda3\envs\jax-env-3.11\python.exe" "%~dp0main.py" smoke
