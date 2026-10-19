#!/usr/bin/env python3
"""Report lexical near-duplicates and template concentration in a task pool.

Rules:
  R1: 4-gram overlap |A intersection B| / min(|A|, |B|) > 0.50.
  R2: term-frequency cosine >= 0.85 (lexical, not semantic embeddings).
  R3: more than 3 task IDs in a heuristically inferred template family.

This script produces a report; it does not remove tasks or compare against a
held-out benchmark. Tokenization uses Unicode word sequences, so languages
without whitespace word boundaries need separate inspection.

Usage:
    python data_construction/internal_dedup.py --pool-dir path/to/tasks
    python data_construction/internal_dedup.py --dataset-dir path/to/tasks --tasks id1 id2
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path(__file__).resolve().parent / "generated_tasks" / "tasks"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

NGRAM_N = 4
OVERLAP_THRESH = 0.50
COSINE_THRESH = 0.85
TEMPLATE_CAP = 3

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _word_4grams(text: str, n: int = NGRAM_N) -> set[tuple[str, ...]]:
    toks = _tokens(text)
    if len(toks) < n:
        return set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def overlap_ratio(a: str, b: str) -> float:
    """Symmetric 4-gram overlap = |A∩B| / min(|A|,|B|).

    Using min() (not union) makes a short prompt fully contained in a long one
    score ~1.0 — the case we most want to catch.
    """
    ga, gb = _word_4grams(a), _word_4grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / min(len(ga), len(gb))


def _tf_vector(text: str) -> Counter:
    return Counter(_tokens(text))


def tf_cosine(a: str, b: str) -> float:
    """Term-frequency cosine (lexical proxy for semantic cosine)."""
    va, vb = _tf_vector(a), _tf_vector(b)
    if not va or not vb:
        return 0.0
    common = set(va) & set(vb)
    dot = sum(va[t] * vb[t] for t in common)
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    return dot / (na * nb) if na and nb else 0.0


def _template_family(task_id: str) -> str:
    """Coarse template key: strip leading index + trailing chart-type token."""
    s = re.sub(r"^[A-Z]\d+_?", "", task_id)  # drop M018_ / T034_
    s = re.sub(r"_(line|bar|radar|scatter|heatmap|pie)(_chart)?$", "", s)
    s = re.sub(r"\d+", "", s)
    return s.strip("_") or task_id


def _load_prompt(task_dir: Path) -> str:
    # Expanded Harbor layout (tests/task.yaml) OR source layout (root task.yaml).
    for rel in ("tests/task.yaml", "task.yaml"):
        task_yaml = task_dir / rel
        if task_yaml.is_file():
            try:
                y = yaml.safe_load(task_yaml.read_text(encoding="utf-8")) or {}
                prompt = y.get("prompt") or {}
                if isinstance(prompt, dict) and prompt.get("text"):
                    return str(prompt["text"])
            except yaml.YAMLError:
                pass
    instr = task_dir / "instruction.md"
    return instr.read_text(encoding="utf-8", errors="ignore") if instr.is_file() else ""


def analyze(prompts: dict[str, str]) -> dict:
    """Pairwise R1/R2 + per-family R3 over {task_id: prompt}."""
    ids = sorted(prompts)
    pairs = []
    for a, b in combinations(ids, 2):
        ov = overlap_ratio(prompts[a], prompts[b])
        cos = tf_cosine(prompts[a], prompts[b])
        if ov > OVERLAP_THRESH or cos >= COSINE_THRESH:
            pairs.append({
                "a": a, "b": b,
                "overlap_4gram": round(ov, 3),
                "cosine_tf": round(cos, 3),
                "rule": "R1_4gram" if ov > OVERLAP_THRESH else "R2_cosine",
            })

    families = Counter(_template_family(i) for i in ids)
    over_cap = {
        fam: cnt for fam, cnt in families.items() if cnt > TEMPLATE_CAP
    }

    return {
        "_meta": {
            "n_tasks": len(ids),
            "n_flagged_pairs": len(pairs),
            "n_families_over_cap": len(over_cap),
            "cosine_mode": "tf_lexical",
            "thresholds": {
                "overlap_4gram": OVERLAP_THRESH,
                "cosine": COSINE_THRESH,
                "template_cap": TEMPLATE_CAP,
            },
        },
        "flagged_pairs": sorted(
            pairs, key=lambda p: max(p["overlap_4gram"], p["cosine_tf"]), reverse=True
        ),
        "families_over_cap": over_cap,
        "family_sizes": dict(families.most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tasks", nargs="+", help="Task ids under --dataset-dir.")
    parser.add_argument("--pool-dir", type=Path, help="Dir of constructed task dirs.")
    parser.add_argument("--candidate", type=Path, help="A single new task dir to add.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "internal_dedup.json")
    args = parser.parse_args()

    prompts: dict[str, str] = {}
    if args.tasks:
        for t in args.tasks:
            prompts[t] = _load_prompt(args.dataset_dir / t)
    if args.pool_dir and args.pool_dir.is_dir():
        for d in sorted(p for p in args.pool_dir.iterdir() if p.is_dir()):
            prompts[d.name] = _load_prompt(d)
    if args.candidate:
        prompts[args.candidate.name] = _load_prompt(args.candidate)

    prompts = {k: v for k, v in prompts.items() if v}
    if len(prompts) < 2:
        logger.error("need >=2 readable prompts, got %d", len(prompts))
        return 1

    report = analyze(prompts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    m = report["_meta"]
    logger.info("analyzed %d tasks → %s", m["n_tasks"], args.output)
    logger.info("cosine mode: %s", m["cosine_mode"])
    logger.info(
        "flagged pairs=%d  families over cap(>%d)=%d",
        m["n_flagged_pairs"], TEMPLATE_CAP, m["n_families_over_cap"],
    )
    for p in report["flagged_pairs"][:5]:
        logger.info("  %s ~ %s  (%s ov=%.2f cos=%.2f)",
                    p["a"], p["b"], p["rule"], p["overlap_4gram"], p["cosine_tf"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
