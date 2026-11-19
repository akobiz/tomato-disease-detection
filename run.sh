#!/usr/bin/env bash
# © 2026 Андрій Кобізь
# Запуск веб-застосунку.
#   ./run.sh         тільки на цьому комп'ютері: http://127.0.0.1:8000/
#   ./run.sh --lan   доступ з телефона в тій самій Wi-Fi (без авторизації, тільки для демонстрації)
set -e
cd "$(dirname "$0")"
[ -d .venv ] || { echo "No .venv found. Follow the Installation section of README.md first."; exit 1; }
source .venv/bin/activate
HOST=127.0.0.1
if [ "$1" = "--lan" ]; then
  HOST=0.0.0.0
  IP=$(ipconfig getifaddr en0 2>/dev/null || true)
  echo "Open on the phone (same Wi-Fi): http://${IP:-<your-Mac-IP>}:8000/"
fi
exec python -m uvicorn src.api.main:app --host "$HOST" --port 8000
