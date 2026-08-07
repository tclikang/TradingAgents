#!/usr/bin/env bash
# ============================================================
# TradingAgents 启动脚本
# 用法:
#   ./run.sh                              # 交互式运行
#   ./run.sh 600519.SS                    # 只指定 ticker，其余交互
#   ./run.sh 600519.SS 2025-01-01 2026-06-30  # 指定全部参数，跳过交互
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 加载 .env ----
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# 清除系统 SOCKS 代理（避免 LLM 客户端报错 Unknown scheme for proxy URL socks://）
export ALL_PROXY=
export all_proxy=

# ---- 环境初始化 ----
echo "> 初始化 conda 环境 py312_dl ..."
eval "$(conda shell.bash hook)"
conda activate py312_dl

echo "> 安装/更新 tradingagents ..."
pip install -q -e "." 2>/dev/null

# ---- 启动 ----
echo ""
echo "============================================"
echo "  TradingAgents - A 股智能交易分析框架"
echo "============================================"
echo ""

tradingagents "$@"
