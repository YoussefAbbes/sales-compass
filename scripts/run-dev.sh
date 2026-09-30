#!/usr/bin/env bash
# Start the dashboard stack on macOS / Linux: FastAPI (fixtures) + Vite React dashboard.
#
#   ./scripts/run-dev.sh        then open http://localhost:5173
#
# The API binds to 127.0.0.1 only and defaults to the committed anonymised fixtures.
# Point MODEL_RESULTS_SOURCE_ROOT at a folder holding real data/ and reports/ to serve
# your own data (that requires MODEL_RESULTS_API_KEY or MODEL_RESULTS_ALLOW_ANONYMOUS=1).
set -euo pipefail
cd "$(dirname "$0")/.."

export MODEL_RESULTS_SOURCE_ROOT="${MODEL_RESULTS_SOURCE_ROOT:-fixtures}"
PORT="${API_PORT:-8001}"

if [ ! -x web/node_modules/.bin/vite ]; then
  echo "Installing frontend dependencies…"
  (cd web && npm install --include=dev)
fi

if (command -v lsof >/dev/null && lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1); then
  echo "Port $PORT is already in use — stop the previous server first." >&2
  exit 1
fi

python3 -m uvicorn model_results_api.app.main:app --host 127.0.0.1 --port "$PORT" &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT INT TERM

echo "API       → http://127.0.0.1:$PORT/docs"
echo "Dashboard → http://localhost:5173"
cd web && npm run dev
