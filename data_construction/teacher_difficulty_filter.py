#!/usr/bin/env python3
"""Summarize observed difficulty from recorded teacher rollout results.

Reads */<task_id>__<trial_suffix>/result.json under one or more job trees.
Historical verdict labels refer to observed mean completion:
    mean >= 0.999: TRIVIAL
    mean <= 0.001: UNSOLVABLE
    otherwise:    KEEP

These labels are sampling heuristics, not evidence that a task is unsolvable or
learnable. Inspect k, variance and infrastructure failures before filtering.
Invalid or missing scores are omitted, not counted as zero. No teacher model
is called by this script and no input files are modified.

Usage:
    python data_construction/teacher_difficulty_filter.py --jobs-dir path/to/jobs \
        --output outputs/teacher_difficulty.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TRIAL_SUFFIX_RE = re.compile(r"__[A-Za-z0-9]+$")
# Historical labels based on mean observed completion, not solvability proofs.
TRIVIAL_HI = 0.999
UNSOLVABLE_LO = 0.001


def _task_id(trial_dir_name: str) -> str:
    return TRIAL_SUFFIX_RE.sub("", trial_dir_name)


def _read_completion(result_json: Path) -> float | None:
    try:
        d = json.loads(result_json.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            return None
        # Harbor per-trial result.json: verifier_result.rewards.{completion,reward}
        vr = d.get("verifier_result")
        if isinstance(vr, dict):
            rewards = vr.get("rewards", vr)  # flat fallback
            if not isinstance(rewards, dict):
                return None
            v = rewards.get("completion", rewards.get("reward"))
        else:
            # Legacy result schema: direct completion/reward field.
            v = d.get("completion", d.get("reward"))
        if v is None or isinstance(v, bool):
            return None
        value = float(v)
        return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


def collect(jobs_dir: Path) -> dict[str, list[float]]:
    """Walk jobs/ -> {task_id: [completion per trial]}."""
    out: dict[str, list[float]] = defaultdict(list)
    # Harbor layout: <jobs_dir>/<task>/<task>__<suffix>/result.json
    for result_json in jobs_dir.rglob("*__*/result.json"):
        tid = _task_id(result_json.parent.name)
        c = _read_completion(result_json)
        if c is not None:
            out[tid].append(c)
    return out


def analyze(scores: dict[str, list[float]]) -> dict[str, Any]:
    tasks = {}
    counts = {"KEEP": 0, "TRIVIAL": 0, "UNSOLVABLE": 0}
    for tid, vals in sorted(scores.items()):
        k = len(vals)
        mean = statistics.fmean(vals) if vals else 0.0
        var = statistics.pvariance(vals) if k >= 2 else 0.0
        if mean >= TRIVIAL_HI:
            verdict = "TRIVIAL"      # high observed mean
        elif mean <= UNSOLVABLE_LO:
            verdict = "UNSOLVABLE"   # low observed mean; inspect failures
        else:
            verdict = "KEEP"         # intermediate mean; review candidate
        counts[verdict] += 1
        tasks[tid] = {
            "k": k,
            "mean_completion": round(mean, 4),
            "min": round(min(vals), 4) if vals else None,
            "max": round(max(vals), 4) if vals else None,
            "variance": round(var, 4),
            "verdict": verdict,
        }
    return {"summary": counts, "tasks": tasks}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs-dir", type=Path, action="append", required=True,
                   help="Job tree(s) of recorded teacher rollouts (repeatable).")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    scores: dict[str, list[float]] = defaultdict(list)
    for jd in args.jobs_dir:
        if not jd.is_dir():
            logger.warning("jobs dir not found, skipping: %s", jd)
            continue
        for tid, vals in collect(jd).items():
            scores[tid].extend(vals)
    if not scores:
        logger.error("no valid completion scores in result.json files under %s", args.jobs_dir)
        return 1

    report = analyze(scores)
    for tid, t in report["tasks"].items():
        logger.info("%-48s k=%d mean=%.3f var=%.3f -> %s",
                    tid[:48], t["k"], t["mean_completion"], t["variance"], t["verdict"])
    s = report["summary"]
    logger.info("KEEP=%d TRIVIAL=%d UNSOLVABLE=%d", s["KEEP"], s["TRIVIAL"], s["UNSOLVABLE"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
