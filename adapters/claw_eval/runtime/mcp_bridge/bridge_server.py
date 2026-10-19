"""Expose declared service tools over MCP and record actual HTTP dispatches.

One bridge process serves one task episode. /dispatches is a read-only evidence
endpoint for the verifier. Keep the sidecar inside an isolated task network;
this is not an authenticated, multi-tenant or internet-facing service.
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any

import httpx
from fastmcp import FastMCP
from fastmcp.tools import Tool
from starlette.responses import JSONResponse
from jsonschema import validate
from jsonschema.exceptions import ValidationError

CONFIG_PATH = Path(os.environ.get("BRIDGE_CONFIG", "/app/bridge_config.json"))

# JSON Schema primitive type → Python type, for signature synthesis.
_JSON_TO_PY: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _make_tool_fn(name: str, description: str, input_schema: dict[str, Any], url: str, method: str, records: list):
    """Build a forwarding coroutine whose signature mirrors the JSON Schema.

    The body accepts **kwargs and POSTs them to the mock; the synthesized
    signature is only there so FastMCP infers the right input schema.
    """

    async def _call(**kwargs: Any) -> str:
        # Drop unset optionals so the mock's pydantic models apply their defaults.
        body = {k: v for k, v in kwargs.items() if v is not None}
        status, result = 599, None
        try:
            validate(body, input_schema)
            async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
                resp = await client.request(method, url, json=body)
                status = resp.status_code
                try:
                    result = resp.json()
                except ValueError:
                    result = resp.text
        except ValidationError:
            status, result = 400, {"error": "Arguments do not match the tool schema"}
        except httpx.HTTPError as exc:
            result = {"error": type(exc).__name__}
        records.append({"tool_name": name, "endpoint_url": url, "request_body": body,
                        "response_status": status, "response_body": result})
        return json.dumps({"status_code": status, "body": result}, ensure_ascii=False)

    properties: dict[str, Any] = (input_schema or {}).get("properties", {}) or {}
    required = set((input_schema or {}).get("required", []) or [])

    params: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}
    for prop_name, prop_spec in properties.items():
        py_type = _JSON_TO_PY.get((prop_spec or {}).get("type", "string"), str)
        annotations[prop_name] = py_type
        if prop_name in required:
            default = inspect.Parameter.empty
        else:
            # Optional → default to the schema default if given, else None.
            default = (prop_spec or {}).get("default", None)
        params.append(
            inspect.Parameter(
                prop_name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=py_type,
            )
        )

    annotations["return"] = str
    _call.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    _call.__annotations__ = annotations
    _call.__name__ = name
    _call.__doc__ = description or name
    return _call


def build_server() -> FastMCP:
    config = _load_config()
    mcp = FastMCP(config.get("server_name", "claweval-bridge"))
    records = []

    @mcp.custom_route("/dispatches", methods=["GET"])
    async def dispatches(request):
        return JSONResponse({"task_id": config["task_id"], "dispatches": list(records)})


    for spec in config["tools"]:
        fn = _make_tool_fn(
            name=spec["name"],
            description=spec.get("description", ""),
            input_schema=spec.get("input_schema", {}),
            url=spec["url"],
            method=spec.get("method", "POST"),
            records=records,
        )
        tool = Tool.from_function(fn, name=spec["name"], description=spec.get("description", ""))
        # Retain nested schema constraints instead of reducing them to Python types.
        tool.parameters = spec.get("input_schema", {"type": "object"})
        mcp.add_tool(tool)
    return mcp


if __name__ == "__main__":
    server = build_server()
    server.run(
        transport="streamable-http",
        host=os.environ.get("BRIDGE_HOST", "0.0.0.0"),
        port=int(os.environ.get("BRIDGE_PORT", "8000")),
    )
