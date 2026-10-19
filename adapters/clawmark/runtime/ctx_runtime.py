"""Build ClawMark's filesystem/email context inside a Harbor task container.

Managers use their retained source APIs over LocalSandbox. Live workspace state
replaces a host-side downloaded snapshot; historical snapshots are unsupported.
This module must run inside the intended task container, not on the host.
"""
from __future__ import annotations
import asyncio
import os
from pathlib import Path
import socket
import sys
import time

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from clawmark_vendor.state.base import BaseStateManager
from clawmark_vendor.context import TaskContext
from clawmark_vendor.sandbox.local import LocalSandbox

WORKSPACE = Path('/workspace')


def _wait_for_email(timeout: float = 60):
    """Wait for actual IMAP/SMTP greetings, not merely a running Java process."""
    deadline = time.monotonic() + timeout
    for host, port, prefix in [
        (os.getenv('EMAIL_SMTP_SERVER', 'localhost'), int(os.getenv('EMAIL_SMTP_PORT', '3025')), b'220'),
        (os.getenv('EMAIL_IMAP_SERVER', 'localhost'), int(os.getenv('EMAIL_IMAP_PORT', '3143')), b'* OK'),
    ]:
        while True:
            try:
                with socket.create_connection((host, port), timeout=2) as conn:
                    conn.settimeout(2)
                    if conn.recv(1024).upper().startswith(prefix):
                        break
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Email backend did not become ready at {host}:{port}')
            time.sleep(.25)


async def build_ctx(task_dir: Path, environments: list[str], env_config: dict | None = None) -> TaskContext:
    if set(environments) - {'filesystem', 'email'}:
        raise ValueError('Only filesystem/email environments are supported')
    if 'email' in environments:
        await asyncio.to_thread(_wait_for_email)
    config = env_config or {}
    sandbox = LocalSandbox()
    await sandbox.start()
    managers = {name: BaseStateManager.create(name, config=config.get(name)) for name in environments}
    for manager in managers.values():
        await manager.setup(sandbox=sandbox)
    ctx = TaskContext(managers=managers, sandbox=sandbox, task_dir=Path(task_dir))
    ctx.workspace = WORKSPACE
    return ctx


def metadata_environments(metadata: dict) -> list[str]:
    return list(metadata.get('environments', []))


def metadata_env_config(metadata: dict) -> dict:
    return dict(metadata.get('env_config', {}) or {})
