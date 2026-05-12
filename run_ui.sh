#!/usr/bin/env bash
set -euo pipefail
cd "/home/diego/backtest historico"
source .venv/bin/activate
python ui_server.py
