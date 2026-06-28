@echo off
chcp 65001 >nul
echo ========================================
echo   TradingAgents - 多Agent股票分析系统
echo ========================================

REM 1. 配置 VPN 代理
echo.
echo [1/3] 配置代理...
set HTTP_PROXY=http://127.0.0.1:7897
set HTTPS_PROXY=http://127.0.0.1:7897
echo   HTTP_PROXY=%HTTP_PROXY%
echo   HTTPS_PROXY=%HTTPS_PROXY%

REM 2. 激活虚拟环境并启动
echo.
echo [2/3] 激活虚拟环境...
call .venv\Scripts\activate.bat
echo   虚拟环境已激活

REM 3. 启动 TradingAgents
echo.
echo [3/3] 启动 TradingAgents...
echo ========================================
echo.

tradingagents %*
