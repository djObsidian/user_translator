@echo off
chcp 65001 >nul
setlocal
set "HERE=%~dp0"
set "PYTHONPATH=%HERE%"
set "PYTHONUTF8=1"

if "%~1"=="" goto usage
"%HERE%.venv\Scripts\python.exe" -m pdftr %*
pause
exit /b

:usage
echo Перетащите один или несколько PDF на этот файл, или запустите из консоли:
echo     translate.bat книга.pdf --to ru
echo.
"%HERE%.venv\Scripts\python.exe" -m pdftr --help
pause
