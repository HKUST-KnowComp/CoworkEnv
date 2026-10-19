"""Claw-Eval single-service Pinbench task adapter for Harbor.

Adapted from the project's existing live MCP path. Source graders are trusted
Python code and are imported during conversion. Only the unchanged
PinbenchAdaptedGrader grading implementation is supported. No synthetic oracle
or agent-authored trace input is emitted.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import yaml

ADAPTER_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = ADAPTER_DIR / "template"
VENDORED_CLAW_EVAL = TEMPLATE_DIR / "tests" / "claw_eval"
MCP_BRIDGE_DIR = ADAPTER_DIR / "runtime" / "mcp_bridge"
LICENSE_PATH = ADAPTER_DIR.parents[1] / "third_party" / "claw-eval.LICENSE"


def _contained_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Expected a relative path inside the source checkout: {relative}")
    result = (root / rel).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes the source checkout: {relative}")
    return result


@dataclass
class ParsedTask:
    source_id: str
    prompt: str
    grader_src: str
    grader_attrs: dict
    category: str
    timeout_seconds: int
    services: list[dict] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)
    tool_endpoints: list[dict] = field(default_factory=list)


def _load_grader_class_from_src(grader_src: str, source_id: str):
    """Exec the grader source (with vendored claw_eval importable) and return its class."""
    if str(VENDORED_CLAW_EVAL.parent) not in sys.path:
        sys.path.insert(0, str(VENDORED_CLAW_EVAL.parent))
    from claw_eval.graders.base import AbstractGrader

    mod_name = f"_gen_grader_{re.sub(r'[^0-9a-zA-Z_]', '_', source_id)}"
    spec = importlib.util.spec_from_loader(mod_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        exec(compile(grader_src, f"<grader:{source_id}>", "exec"), module.__dict__)
    except ModuleNotFoundError as e:
        # Graders that import claw_eval modules we deliberately did NOT vendor
        # (e.g. image_qa_oracle / llm_judge) are out of the rule-based-text scope.
        raise UnsupportedTaskError(
            f"grader imports a non-vendored module ({e.name}); out of text-rule scope"
        ) from e
    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(obj, AbstractGrader)
            and obj is not AbstractGrader
            and obj.__module__ == module.__name__
        ):
            return obj
    raise ValueError(f"No AbstractGrader subclass in grader.py for {source_id}")


def parse_task(task_dir: Path) -> ParsedTask:
    yaml_path = task_dir / "task.yaml"
    grader_path = task_dir / "grader.py"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    grader_src = grader_path.read_text(encoding="utf-8")

    source_id = data.get("task_id", task_dir.name)
    if not isinstance(source_id, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", source_id):
        raise ValueError("Invalid task_id")
    grader_cls = _load_grader_class_from_src(grader_src, source_id)

    attrs = {
        "REQUIRED_TOOLS": dict(getattr(grader_cls, "REQUIRED_TOOLS", {}) or {}),
        "FORBIDDEN_TOOLS": list(getattr(grader_cls, "FORBIDDEN_TOOLS", []) or []),
        "REQUIRED_ACTIONS": list(getattr(grader_cls, "REQUIRED_ACTIONS", []) or []),
        "REQUIRED_KEYWORDS": list(getattr(grader_cls, "REQUIRED_KEYWORDS", []) or []),
        "OPTIONAL_KEYWORDS": list(getattr(grader_cls, "OPTIONAL_KEYWORDS", []) or []),
        "REQUIRED_PATTERNS": list(getattr(grader_cls, "REQUIRED_PATTERNS", []) or []),
        "MIN_FINAL_LENGTH": int(getattr(grader_cls, "MIN_FINAL_LENGTH", 0) or 0),
        "REQUIRE_IMAGE_LOAD": bool(getattr(grader_cls, "REQUIRE_IMAGE_LOAD", False)),
    }

    from claw_eval.graders.pinbench_common import PinbenchAdaptedGrader
    if not issubclass(grader_cls, PinbenchAdaptedGrader) or grader_cls.grade is not PinbenchAdaptedGrader.grade:
        raise UnsupportedTaskError("Only PinbenchAdaptedGrader subclasses with unchanged grade() are supported")
    if grader_cls._get_final_assistant_text is not PinbenchAdaptedGrader._get_final_assistant_text or grader_cls.compute_robustness is not PinbenchAdaptedGrader.compute_robustness:
        raise UnsupportedTaskError("Custom grading helpers are not supported")
    if attrs["REQUIRED_ACTIONS"] or attrs["REQUIRE_IMAGE_LOAD"]:
        raise UnsupportedTaskError("Service-state and media grading require a separate adapter")
    if len(data.get("services", [])) != 1:
        raise UnsupportedTaskError("This release supports exactly one service per task")
    prompt = (data.get("prompt") or {}).get("text", "").strip()
    if not prompt:
        raise ValueError("Task prompt is empty")
    return ParsedTask(
        source_id=source_id,
        prompt=prompt,
        grader_src=grader_src,
        grader_attrs=attrs,
        category=re.sub(r"[^A-Za-z0-9_-]", "_", str(data.get("category", "research"))),
        timeout_seconds=int((data.get("environment") or {}).get("timeout_seconds", 180)),
        services=list(data.get("services", []) or []),
        tools=list(data.get("tools", []) or []),
        tool_endpoints=list(data.get("tool_endpoints", []) or []),
    )


class UnsupportedTaskError(Exception):
    """Task requires capabilities outside this adapter's scope."""


LIVE_INSTRUCTION_SUFFIX = (
    "\n\nThe tools you need are available over MCP (server `claweval-tools`). "
    "Call them to gather the information required, then write your final answer "
    "as your last message. Your tool usage and final answer are graded "
    "automatically — you do NOT need to write any trace file.\n"
)


TASK_TOML_LIVE = """\
version = "1.0"

[task]
name = "claw-eval/{source_id}"
authors = []
keywords = ["claw-eval", "pinbench", "rule-based", "trace-graded", "live", "mcp"]

[metadata]
author_name = "claw-eval"
author_email = ""
difficulty = "easy"
category = "{category}"
tags = ["claw-eval", "pinbench", "rule-based", "trace-graded", "live", "mcp"]
source_id = "{source_id}"
grading = "rule"
variant = "live"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = {agent_timeout:.1f}

[environment]
build_timeout_sec = 900.0
cpus = 2
memory_mb = 4096
storage_mb = 4096
gpus = 0
network_mode = "public"

[[environment.mcp_servers]]
name = "claweval-tools"
transport = "streamable-http"
url = "{mcp_url}"

[verifier.env]

[solution.env]
"""


DOCKERFILE_LIVE = """\
FROM node:20-alpine

# claw-eval LIVE task ({source_id}). The agent (claude-code) is installed by
# Harbor; we only need bash + python for the verifier's vendored grader
# (pydantic + pyyaml) and bridge evidence retrieval (stdlib urllib).
RUN apk add --no-cache bash python3 py3-pip curl \\
    && python3 -m pip install --no-cache-dir --break-system-packages "pydantic==2.14.1" "pyyaml==6.0.3"

ENV TMP_WORKSPACE=/workspace
RUN mkdir -p /workspace
WORKDIR /workspace
CMD ["/bin/bash"]
"""


COMPOSE_LIVE = """\
# Merged over Harbor's base compose. `claweval-tools` co-locates the claw-eval
# mock service ({mock_module} on :{mock_port}) and the MCP→HTTP bridge (:8000).
# The agent reaches tools at http://claweval-tools:8000/mcp; the verifier reads
# dispatch evidence at http://claweval-tools:8000/dispatches.
services:
  main:
    depends_on:
      claweval-tools:
        condition: service_healthy

  claweval-tools:
    build:
      context: ./mcp-bridge
    hostname: claweval-tools
    environment:
      - MOCK_MODULE={mock_module}
      - MOCK_PORT={mock_port}
      - BRIDGE_PORT=8000
      - ERROR_RATE=0
{fixtures_env}    expose:
      - "8000"
      - "{mock_port}"
    healthcheck:
      test: ["CMD", "python", "-c", "import socket; s=socket.create_connection(('localhost',8000),timeout=2); s.close()"]
      interval: 3s
      timeout: 5s
      retries: 20
      start_period: 8s
"""


def _module_from_command(command: str) -> str:
    """'python mock_services/rss/server.py' -> 'mock_services.rss.server'."""
    parts = command.split()
    script = parts[-1] if parts else command
    return script.removesuffix(".py").replace("/", ".")


def build_live_payload(parsed: ParsedTask, out: Path, source_root: Path) -> dict:
    """Assemble out/environment/mcp-bridge/ and return compose + live params.

    Supports the single-service pinbench tasks (T087/T090/T095). Copies the
    selected claw-eval service and shared helpers + the task's fixtures, plus the
    bridge server, and writes bridge_config.json (tool→endpoint map for the MCP
    bridge) and live_config.json (bridge evidence URL for the verifier).
    """
    if len(parsed.services) != 1:
        raise UnsupportedTaskError(
            f"live variant supports exactly 1 service; {parsed.source_id} has "
            f"{len(parsed.services)}"
        )
    service = parsed.services[0]
    if not 1 <= int(service["port"]) <= 65535:
        raise ValueError("Invalid service port")
    names = [tool["name"] for tool in parsed.tools]
    endpoints = [item["tool_name"] for item in parsed.tool_endpoints]
    if not names or len(set(names)) != len(names) or len(set(endpoints)) != len(endpoints) or set(names) != set(endpoints):
        raise ValueError("Tool definitions and endpoints must match one-to-one")
    bridge_dst = out / "environment" / "mcp-bridge"
    bridge_dst.mkdir(parents=True, exist_ok=True)

    # Bridge server + Dockerfile + start.sh (verbatim from runtime/mcp_bridge).
    for fname in ("bridge_server.py", "Dockerfile", "start.sh", "requirements.txt"):
        shutil.copy(MCP_BRIDGE_DIR / fname, bridge_dst / fname)

    # Copy only the selected service and shared package-level Python helpers.
    module = _module_from_command(service["command"])
    if not re.fullmatch(r"mock_services\.[a-z_]+\.server", module):
        raise UnsupportedTaskError("Expected python mock_services/<service>/server.py")
    msrc = _contained_path(source_root, "mock_services")
    mdst = bridge_dst / "mock_services"
    if mdst.exists():
        shutil.rmtree(mdst)
    mdst.mkdir()
    service_dir = _contained_path(msrc, module.split(".")[1])
    for src in list(msrc.glob("*.py")) + list(service_dir.rglob("*.py")):
        if src.is_symlink():
            raise ValueError("Service source symlinks are not supported")
        dest = mdst / src.relative_to(msrc)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    if not (mdst / module.split(".")[1] / "server.py").is_file():
        raise FileNotFoundError("Service server.py is missing")
    shutil.copy2(LICENSE_PATH, mdst / "LICENSE")

    # Copy the task's fixtures (referenced by service env, paths relative to
    # claw-eval root) into the bridge image, and rewrite env to the in-image path.
    fixtures_env_lines: list[str] = []
    for env_key, rel_path in (service.get("env") or {}).items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*_FIXTURES", env_key):
            raise UnsupportedTaskError("Only file-valued *_FIXTURES service variables are supported")
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", rel_path):
            raise ValueError("Invalid fixture path")
        src_fix = _contained_path(source_root, rel_path)
        dst_fix = bridge_dst / "fixtures" / rel_path
        dst_fix.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_fix, dst_fix)
        fixtures_env_lines.append(f"      - {env_key}=/app/fixtures/{rel_path}\n")
    # The Dockerfile COPYs fixtures/ unconditionally; ensure the dir exists.
    (bridge_dst / "fixtures").mkdir(exist_ok=True)

    # bridge_config.json: tool name + schema + endpoint URL (verbatim from yaml).
    endpoint_by_tool = {te["tool_name"]: te for te in parsed.tool_endpoints}
    schema_by_tool = {t["name"]: t.get("input_schema", {}) for t in parsed.tools}
    bridge_tools = []
    for t in parsed.tools:
        te = endpoint_by_tool.get(t["name"])
        if te is None:
            raise UnsupportedTaskError(f"No endpoint for tool {t['name']}")
        endpoint = urlsplit(te["url"])
        if endpoint.scheme != "http" or endpoint.hostname not in {"localhost", "127.0.0.1"} or endpoint.port != int(service["port"]) or endpoint.username or endpoint.password:
            raise UnsupportedTaskError("Tool endpoints must use the declared local service port")
        bridge_tools.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": schema_by_tool.get(t["name"], {}),
            "url": te["url"],
            "method": te.get("method", "POST"),
        })
    (bridge_dst / "bridge_config.json").write_text(
        json.dumps({"server_name": "claweval-tools", "task_id": parsed.source_id, "tools": bridge_tools}, indent=2),
        encoding="utf-8",
    )

    # live_config.json points to bridge-recorded dispatch evidence.
    mock_port = int(service["port"])
    live_config = {
        "task_id": parsed.source_id,
        "dispatch_url": "http://claweval-tools:8000/dispatches",
        "tool_names": [t["name"] for t in parsed.tools],
    }

    return {
        "mock_module": _module_from_command(service["command"]),
        "mock_port": mock_port,
        "fixtures_env": "".join(fixtures_env_lines),
        "live_config": live_config,
    }


class ClawEvalAdapter:
    NAME = "claw-eval"

    def __init__(self, out_dir: Path, source_root: Path):
        self.out_dir = Path(out_dir)
        self.source_root = Path(source_root).resolve(strict=True)

    def generate_task(self, task_dir: Path) -> Path:
        parsed = parse_task(task_dir)
        out = self.out_dir / parsed.source_id
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite {out}")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".convert-", dir=self.out_dir) as tmp:
            stage = Path(tmp)
            live = build_live_payload(parsed, stage, self.source_root)
            tests = stage / "tests"
            tests.mkdir()
            (stage / "instruction.md").write_text(parsed.prompt + LIVE_INSTRUCTION_SUFFIX)
            (stage / "task.toml").write_text(TASK_TOML_LIVE.format(
                source_id=parsed.source_id, category=parsed.category.replace('"', ''),
                agent_timeout=float(parsed.timeout_seconds), mcp_url="http://claweval-tools:8000/mcp"))
            (stage / "environment/Dockerfile").write_text(DOCKERFILE_LIVE.format(source_id=parsed.source_id))
            (stage / "environment/docker-compose.yaml").write_text(COMPOSE_LIVE.format(**live))
            for name in ("test_runner.py", "test.sh"):
                shutil.copy2(TEMPLATE_DIR / "tests" / name, tests / name)
            shutil.copytree(VENDORED_CLAW_EVAL, tests / "claw_eval", ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copy2(LICENSE_PATH, tests / "claw_eval/LICENSE")
            (tests / "grader.py").write_text(parsed.grader_src)
            (tests / "live_config.json").write_text(json.dumps(live["live_config"], indent=2))
            (tests / "test.sh").chmod(0o755)
            shutil.move(str(stage), out)
        return out
