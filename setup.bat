@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PY=python"
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"

if not exist .venv\Scripts\python.exe (
    echo [setup] создаю окружение .venv
    %PY% -m venv .venv
    if errorlevel 1 goto nopython
)
.venv\Scripts\python -c "import sys; sys.exit(sys.version_info < (3, 11))"
if errorlevel 1 goto nopython

echo [setup] ставлю зависимости
.venv\Scripts\python -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 goto fail

echo [setup] качаю llama.cpp и модели, это десятки гигабайт, можно прервать и перезапустить
.venv\Scripts\python setup_models.py %*
if errorlevel 1 goto fail

echo.
echo Установка завершена. Теперь перетаскивайте PDF на translate.bat
pause
exit /b 0

:nopython
echo Нужен Python 3.11 или новее: https://www.python.org/downloads/ - при установке отметьте "Add python.exe to PATH".
pause
exit /b 1

:fail
echo Ошибка установки, смотрите сообщения выше.
pause
exit /b 1
