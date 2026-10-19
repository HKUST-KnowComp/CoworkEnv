FROM python:3.12-slim

# claw-eval LIVE sidecar: runs ONE mock service (rss/kb/config/...) plus the
# generic MCP→HTTP bridge in a single container. The agent (in the `main`
# container) reaches the bridge over the compose network via streamable-http at
# http://claweval-tools:8000/mcp; the bridge forwards each tool call to the mock
# at localhost:<port> (so task.yaml's verbatim tool_endpoints URLs work as-is),
# and the Harbor verifier reads actual statuses from the bridge's /dispatches.
#
# Co-locating mock + bridge keeps task.yaml's http://localhost:<port>/... URLs
# unchanged and gives the verifier a single host to query for the audit log.

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# The selected service and shared package helpers are copied from the user's
# upstream checkout at conversion time, along with the selected task fixtures.
COPY mock_services/ /app/mock_services/
COPY fixtures/ /app/fixtures/
COPY bridge_server.py /app/bridge_server.py
COPY bridge_config.json /app/bridge_config.json
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# MOCK_CMD / MOCK_PORT / fixtures env are injected per-task by docker-compose.
EXPOSE 8000
CMD ["/app/start.sh"]
