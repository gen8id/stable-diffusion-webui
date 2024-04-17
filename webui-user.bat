@echo off

:: set PYTHON=
:: set GIT=
:: set VENV_DIR=
set COMMANDLINE_ARGS=--api --skip-torch-cuda-test --xformers

call webui.bat
