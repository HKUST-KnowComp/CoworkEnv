#!/bin/bash
# claw-eval live sidecar entrypoint: start the mock service, wait for it to be
# healthy, then start the MCP bridge in the foreground (PID 1 for the container).
#
# Injected by docker-compose (per task):
#   MOCK_MODULE   e.g. mock_services.rss.server   (run with python -m)
#   MOCK_PORT     e.g. 9109
#   <FIXTURES>    e.g. RSS_FIXTURES=/app/fixtures/rss/articles.json
#   BRIDGE_PORT   default 8000
#   ERROR_RATE    default 0 (deterministic; the grader counts response_status<400)
set -euo pipefail

MOCK_MODULE="${MOCK_MODULE:?MOCK_MODULE must be set (e.g. mock_services.rss.server)}"
MOCK_PORT="${MOCK_PORT:?MOCK_PORT must be set (e.g. 9109)}"
BRIDGE_PORT="${BRIDGE_PORT:-8000}"
export ERROR_RATE="${ERROR_RATE:-0}"
export PORT="$MOCK_PORT"

echo "[sidecar] starting mock $MOCK_MODULE on :$MOCK_PORT (ERROR_RATE=$ERROR_RATE)"
python -m "$MOCK_MODULE" &
MOCK_PID=$!

# Wait for the mock to accept connections before the bridge advertises its tools.
ready=0
for i in $(seq 1 30); do
    if curl -fsS -o /dev/null "http://localhost:${MOCK_PORT}/docs" 2>/dev/null; then
        echo "[sidecar] mock healthy after ${i} tries"
        ready=1
        break
    fi
    if ! kill -0 "$MOCK_PID" 2>/dev/null; then
        echo "[sidecar] mock process died during startup" >&2
        exit 1
    fi
    sleep 1
done

if [ "$ready" -ne 1 ]; then
    echo "[sidecar] service readiness timed out" >&2
    kill "$MOCK_PID" 2>/dev/null || true
    exit 1
fi

echo "[sidecar] starting MCP bridge on :$BRIDGE_PORT -> mock :$MOCK_PORT"
exec python /app/bridge_server.py
