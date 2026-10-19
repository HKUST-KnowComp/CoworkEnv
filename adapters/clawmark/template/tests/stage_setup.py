#!/usr/bin/env python3
"""Per-step setup: run a ClawMark stage's SEEDING function via the in-container ctx.

Usage: python3 stage_setup.py <stage_id>

Each ClawMark stageN(ctx) both seeds state (uploads assets, sends emails, writes
files) AND returns a notification dict. In Harbor we run the seeding here, before
the agent acts, so the environment matches what the source harness would present.
The notification text itself is baked into the step's instruction.md at generation
time (it is a static string), so we only need the side effects here.

The ``ctx`` is built through ClawMark's own code path (ctx_runtime.build_ctx),
using the task's METADATA. The adapter treats failed setup shell commands or
failed email sends as errors rather than continuing with incomplete state.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ctx_runtime import build_ctx, metadata_env_config, metadata_environments  # noqa: E402


def _load_task_module():
    spec = importlib.util.spec_from_file_location("clawmark_task", HERE / "clawmark_task.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


async def _seed(stage_id: str) -> None:
    module = _load_task_module()
    stage_fn = getattr(module, stage_id, None)
    if stage_fn is None:
        print(f"[setup] no {stage_id}() in task.py; nothing to seed.")
        return

    metadata = getattr(module, "METADATA", {})
    ctx = await build_ctx(
        task_dir=HERE,
        environments=metadata_environments(metadata),
        env_config=metadata_env_config(metadata),
    )
    execute = ctx.sandbox.exec

    async def checked_exec(*args, **kwargs):
        result = await execute(*args, **kwargs)
        if result.return_code != 0:
            raise RuntimeError(f'Seeding shell command failed (exit {result.return_code})')
        return result

    ctx.sandbox.exec = checked_exec
    if hasattr(ctx, 'email'):
        send = ctx.email.send_email

        async def checked_send(*args, **kwargs):
            if not await send(*args, **kwargs):
                raise RuntimeError('Seeding email delivery failed')
            return True

        ctx.email.send_email = checked_send
    # Failure must abort setup; training on a partially seeded state is invalid.
    ret = await stage_fn(ctx)
    print(f"[setup] ran {stage_id}() seeding; notification keys: "
          f"{list(ret.keys()) if isinstance(ret, dict) else type(ret).__name__}")


def main() -> int:
    stage_id = sys.argv[1] if len(sys.argv) > 1 else "stage0"
    asyncio.run(_seed(stage_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
