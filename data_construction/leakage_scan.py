#!/usr/bin/env python3
"""Scan task text for explicitly supplied numeric ground-truth values.

Checks instruction.md, task.yaml prompt/reference_solution, and recognized text
files under exec/ and fixtures/. This is a heuristic for known layouts, not a
proof of isolation or absence of leakage. Binary files are not inspected.

Matches unsigned decimal tokens, including trailing zeros and percent signs,
with tolerance max(1e-6, abs(gt) * 1e-4). Integer values may be explicitly
ignored; the CLI defaults to ignoring 0 and 100. Negative and scientific
notation are not supported reliably by this matcher.

Usage:
    python data_construction/leakage_scan.py --task-dir path/to/task --gt 69.6 49.7
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EPS = 1e-6
# A numeric token: optional decimal, optional trailing %. Left boundary rejects
# a preceding word char or dot (so 45.1 isn't the tail of v45 or x.45.1). Right
# boundary rejects a following word char and rejects ".<digit>" (so 45 in 45.1
# isn't matched, and 45.1 isn't part of 45.12) — but a sentence-final period
# (no digit after) is allowed, fixing values like "...scored 45.1.".
_NUM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?:\s*%)?(?![\w])(?!\.\d)")


@dataclass
class Leak:
    source: str   # which agent-visible surface leaked
    gt_value: float
    matched_text: str
    line: int


def _numbers_in(text: str) -> list[tuple[float, str, int]]:
    """Every numeric token with its raw text and 1-based line number."""
    out: list[tuple[float, str, int]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for m in _NUM_RE.finditer(line):
            try:
                out.append((float(m.group(1)), m.group(0), lineno))
            except ValueError:
                pass
    return out


def scan_text(
    text: str,
    source: str,
    gt_values: list[float],
    ignore_ints: set[int] | None = None,
) -> list[Leak]:
    """Return every GT value that appears as a number in `text`.

    ignore_ints whitelists integer GT values that legitimately appear in chart
    prompts (axis bounds, counts). Decimal GT values are never whitelisted.
    """
    ignore_ints = ignore_ints or set()
    found_numbers = _numbers_in(text)
    leaks: list[Leak] = []
    for gt in gt_values:
        is_int = abs(gt - round(gt)) < EPS
        if is_int and int(round(gt)) in ignore_ints:
            continue
        for value, raw, lineno in found_numbers:
            if abs(value - gt) < max(EPS, abs(gt) * 1e-4):
                leaks.append(Leak(source, gt, raw, lineno))
                break  # one hit per GT per source is enough to flag
    return leaks


def _agent_visible_surfaces(task_dir: Path) -> dict[str, str]:
    """Collect known text surfaces; callers must check their runtime's exposure."""
    surfaces: dict[str, str] = {}

    instr = task_dir / "instruction.md"
    if instr.is_file():
        surfaces["instruction.md"] = instr.read_text(encoding="utf-8")

    task_yaml = task_dir / "tests" / "task.yaml"
    src_yaml = task_dir / "task.yaml"
    yaml_path = task_yaml if task_yaml.is_file() else src_yaml
    if yaml_path.is_file():
        y = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(y, dict):
            raise ValueError(f"{yaml_path}: expected a task mapping")
        prompt = y.get("prompt") or {}
        ptext = prompt.get("text", "") if isinstance(prompt, dict) else str(prompt)
        surfaces["task.yaml:prompt"] = str(ptext)
        # Some adapters expose this field in the agent instruction.
        surfaces["task.yaml:reference_solution"] = str(y.get("reference_solution", ""))

    # Binary fixtures require separate review; this scanner only reads text.
    for fx in list((task_dir / "exec").rglob("*")) + list((task_dir / "fixtures").rglob("*")):
        if fx.is_file() and fx.suffix.lower() in {".json", ".csv", ".txt", ".md", ".yaml", ".yml"}:
            surfaces[f"fixture:{fx.relative_to(task_dir)}"] = fx.read_text(encoding="utf-8")
    return surfaces


def check_task_dir(
    task_dir: Path, gt_values: list[float], ignore_ints: set[int] | None = None
) -> list[Leak]:
    """Scan known text surfaces. Raise on missing/unreadable inputs."""
    if not task_dir.is_dir():
        raise ValueError(f"task dir not found: {task_dir}")
    if not gt_values or not all(math.isfinite(v) for v in gt_values):
        raise ValueError("ground-truth values must be non-empty and finite")
    surfaces = _agent_visible_surfaces(task_dir)
    if not any(text.strip() for text in surfaces.values()):
        raise ValueError(f"no scannable text found: {task_dir}")
    leaks: list[Leak] = []
    for source, text in surfaces.items():
        if text.strip():
            leaks.extend(scan_text(text, source, gt_values, ignore_ints))
    return leaks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument(
        "--gt", nargs="+", type=float, required=True,
        help="Ground-truth values that must NOT appear in agent-visible text.",
    )
    parser.add_argument(
        "--ignore-ints", nargs="*", type=int, default=[0, 100],
        help="Integer GT values allowed in prompts (axis bounds etc.); "
        "default 0 100. Decimals are never ignored.",
    )
    args = parser.parse_args()

    if not args.task_dir.is_dir():
        logger.error("task dir not found: %s", args.task_dir)
        return 1

    try:
        leaks = check_task_dir(args.task_dir, args.gt, set(args.ignore_ints))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        logger.error("scan incomplete: %s", exc)
        return 1
    if not leaks:
        logger.info("No supplied GT values matched in scanned text of %s", args.task_dir.name)
        return 0

    logger.error("LEAK — %d GT value(s) reachable by the agent in %s:", len(leaks), args.task_dir.name)
    for lk in leaks:
        logger.error("  GT %.4g leaked in %s @ L%d as `%s`", lk.gt_value, lk.source, lk.line, lk.matched_text)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
