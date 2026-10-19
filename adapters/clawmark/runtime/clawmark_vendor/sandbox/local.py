"""In-container transport implementing ClawMark's BaseSandbox interface.

Uses local subprocesses and file copies instead of Docker exec/cp. It does not
create an isolation boundary and must be used inside the intended task container.
Directory creation uses pathlib so destination paths are not shell-interpolated.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

from .base import BaseSandbox, ExecResult


class LocalSandbox(BaseSandbox):
    """In-container BaseSandbox: subprocess for exec, local fs for transfers."""

    # No dynamic port remapping in-container: greenmail is reached by hostname
    # over the compose network at its real ports, so managers use their defaults.
    ports: dict[int, int] = {}

    async def start(self) -> None:
        # The container is already running (we execute inside it); nothing to do.
        await self.exec("mkdir -p /workspace")

    async def stop(self, delete: bool = True) -> None:
        # Lifecycle owned by Harbor; the sandbox does not tear down its own host.
        return None

    async def exec(
        self, command: str, timeout_sec: int = 300, env: dict[str, str] | None = None
    ) -> ExecResult:
        # Source runs `bash -c <command>` in the main container; we do the same
        # locally. An explicit bash gives the same word-splitting
        # / globbing semantics to source's `bash -c`.
        proc = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env={**os.environ, **(env or {})},
        )
        return ExecResult(stdout=proc.stdout, stderr=proc.stderr, return_code=proc.returncode)

    async def upload_file(self, local_path: Path, remote_path: str) -> None:
        dst = Path(remote_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dst)

    async def upload_dir(self, local_dir: Path, remote_dir: str) -> None:
        # Source: `mkdir -p remote` then `docker cp local/. container:remote`
        # (copies CONTENTS of local_dir into remote_dir).
        src = Path(local_dir)
        dst = Path(remote_dir)
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            target = dst / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

    async def download_file(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote_path, local_path)

    async def download_dir(self, remote_dir: str, local_dir: Path) -> None:
        # Source copies CONTENTS of remote_dir into local_dir.
        local_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(remote_dir, local_dir, dirs_exist_ok=True)
