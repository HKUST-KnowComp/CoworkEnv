#!/usr/bin/env python3
"""Stage-parameterized Harbor verifier for a generated ClawMark task.

Usage: python3 test_runner.py <stage_id>     # stage0 | stage1 | ... | final

Imports the VERBATIM source task.py (clawmark_task.py) and runs RUBRIC[<stage_id>]
checkers against a ``ctx`` built through ClawMark's OWN code path (ctx_runtime.
build_ctx → CompositeStateManager-style managers + TaskContext over a LocalSandbox).
Score = passed_weight / total_weight, exactly ClawMark Orchestrator._run_checkers.
Writes {"reward": score}.

``environments`` and ``env_config`` come straight from the task's verbatim
METADATA, so ctx.fs / ctx.email are real source manager instances.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
VERIFIER_DIR = Path("/logs/verifier")
sys.path.insert(0, str(TESTS_DIR))

from ctx_runtime import build_ctx, metadata_env_config, metadata_environments  # noqa: E402


def _load_task_module():
    spec = importlib.util.spec_from_file_location("clawmark_task", TESTS_DIR / "clawmark_task.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_reward(value: float) -> None:
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    (VERIFIER_DIR / "reward.json").write_text(json.dumps({"reward": float(value)}), encoding="utf-8")
    print(f"[verifier] wrote reward.json: reward={float(value)}")


async def _run(entries, module):
    if not isinstance(entries, list) or not entries:
        raise ValueError('Rubric stage must contain checker entries')
    weights = [float(entry.get('weight', 1.0)) for entry in entries]
    if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError('Rubric weights must be finite/nonnegative with positive total')
    # Build ctx through the source code path, using the task's own METADATA.
    metadata = getattr(module, "METADATA", {})
    ctx = await build_ctx(
        task_dir=TESTS_DIR,
        environments=metadata_environments(metadata),
        env_config=metadata_env_config(metadata),
    )
    # == Orchestrator._run_checkers : weighted pass-rate over the rubric entries.
    results = []
    for entry in entries:
        checker = entry["checker"]
        weight = float(entry.get("weight", 1.0))
        try:
            passed = bool(await checker(ctx))
        except Exception as exc:  # a crashing checker counts as not-passed
            print(f"[verifier] checker {entry.get('id')} raised: {exc}", file=sys.stderr)
            passed = False
        results.append((entry.get("id"), passed, weight))
    total = sum(w for _, _, w in results)
    passed_w = sum(w for _, p, w in results if p)
    return (passed_w / total if total > 0 else 0.0), results


def main() -> int:
    stage_id = sys.argv[1] if len(sys.argv) > 1 else "stage0"
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    (VERIFIER_DIR / 'reward.json').unlink(missing_ok=True)
    (VERIFIER_DIR / 'error.json').unlink(missing_ok=True)
    (VERIFIER_DIR / 'checker_results.json').unlink(missing_ok=True)
    try:
        module = _load_task_module()
        entries = getattr(module, "RUBRIC")[stage_id]
        score, results = asyncio.run(_run(entries, module))
    except Exception as exc:
        print(f"[verifier] invalid task/runtime for {stage_id}: {exc}", file=sys.stderr)
        (VERIFIER_DIR / 'error.json').write_text(json.dumps({'stage': stage_id, 'error': str(exc)}))
        return 1
    print(f"[verifier] {stage_id} checker results (weighted pass-rate):")
    for cid, passed, weight in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] w={weight:<4} {cid}")
    print(f"[verifier] {stage_id} score = {score:.4f}")
    _write_reward(score)
    (VERIFIER_DIR / 'checker_results.json').write_text(json.dumps([
        {'id': cid, 'passed': passed, 'weight': weight} for cid, passed, weight in results
    ], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
