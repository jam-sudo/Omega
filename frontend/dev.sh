#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "Starting Omega backend on :8000..."
source .venv/bin/activate
omega serve start --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "Starting Vite dev server on :5173..."
cd frontend
npm run dev &
FRONTEND_PID=$!
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
echo "Backend PID=$BACKEND_PID, Frontend PID=$FRONTEND_PID"
echo "Open http://localhost:5173"
wait
