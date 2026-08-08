@echo off
chcp 65001 >nul 2>&1
echo ========================================
echo   TradingAgents - duo Agent gu piao fen xi
echo ========================================
echo.

if exist ".env" for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do set "%%a=%%b"

echo [1/2] Activating venv...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] .venv not found: python -m venv .venv
    pause
    exit /b 1
)

echo [2/2] Installing tradingagents...
pip install -q -e . 2>nul

echo.
echo ========================================
echo   Starting TradingAgents...
echo ========================================
echo.
tradingagents %*
