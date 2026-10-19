#!/usr/bin/env python3
"""Export Harbor rollout trials into a single training-data JSONL.

This is a THIN wrapper over Harbor's own `harbor.utils.traces_utils.export_traces`
(the same engine behind `harbor traces export`). That function already parses each
trial's ATIF `agent/trajectory.json` into a row with:
  - conversations : list[{"role", "content"}]  (the full agent dialogue)
  - result        : the trial reward as a string (e.g. "0.6") or an exception name
  - instruction   : the task prompt (when --include-instruction)
  - task / model / agent / trial_name / run_id / episode / date

This wrapper writes local JSONL; it never pushes to the Hugging Face Hub.
It reports skipped trials, validates main-agent ATIF input and refuses partial
exports on conversion errors unless --allow-partial is explicitly supplied.
Rows are conversation prefixes, not necessarily complete task trajectories;
use --episodes last for one final prefix per trial. Subagent export is excluded.

Examples:
    # no-cost sanity check on existing trajectories
    python scripts/export_rollouts.py --jobs-dir jobs/my_run --dry-run

    # export every rollout under jobs/ to one JSONL
    python scripts/export_rollouts.py

    # keep only successful (reward >= 0.999) rollouts for filtered-SFT
    python scripts/export_rollouts.py --min-reward 0.999 --output datasets/rollouts/sft.jsonl
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export Harbor rollout trials to a training-data JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--jobs-dir", type=Path, default=PROJECT_ROOT / "jobs",
                   help="Trial dir or a root containing trials recursively (default: jobs/)")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "rollouts.jsonl",
                   help="Output JSONL path (default: outputs/rollouts.jsonl)")
    p.add_argument("--filter", dest="success_filter", choices=["success", "failure", "all"],
                   default="all", help="Filter trials by result (default: all)")
    p.add_argument("--episodes", choices=["all", "last"], default="all",
                   help="Export every episode per trial, or only the last (default: all)")
    p.add_argument("--no-instruction", dest="include_instruction", action="store_false",
                   help="Omit the task instruction column (included by default)")
    p.add_argument("--min-reward", type=float, default=None,
                   help="Drop rows whose numeric reward is below this threshold (for filtered-SFT)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report discovery + reward stats without writing the JSONL")
    p.add_argument("--allow-partial", action="store_true",
                   help="Explicitly allow export despite invalid or unconvertible trials")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    args = p.parse_args(argv)
    if args.min_reward is not None and not math.isfinite(args.min_reward):
        p.error("--min-reward must be finite")
    return args


def _agent_supports_atif(agent_name: str) -> bool:
    """Only agents that emit ATIF agent/trajectory.json can be exported.

    `oracle`/`nop` (and any non-LLM agent) don't, and `export_traces` *raises* on
    them rather than skipping — so we must pre-filter trials before exporting.
    """
    from harbor.agents.factory import AgentFactory
    from harbor.models.agent.name import AgentName
    try:
        agent_class = AgentFactory.get_agent_class(AgentName(agent_name))
    except ValueError:
        return False
    return bool(agent_class is not None and agent_class.capabilities.atif)


def _has_trajectory(trial_dir: Path) -> bool:
    """True if the trial actually wrote an agent/trajectory.json.

    A claude-code trial that errored before producing output (e.g. CLI install
    failed) supports ATIF but has no trajectory file — it must not be counted as
    exportable.
    """
    return (trial_dir / "agent" / "trajectory.json").is_file()


def _discover_exportable_trials(jobs_dir: Path):
    """Return (exportable_trial_dirs, n_total_trials, metadata_failures).

    Walks Harbor's own trial discovery, keeping only trials whose agent supports
    ATIF *and* that actually wrote a trajectory.json. This lets a mixed jobs/ dir
    (oracle + nop + errored + good claude-code runs) be exported without the
    NotImplementedError oracle/nop would raise, and without counting errored runs.
    """
    from harbor.utils.traces_utils import iter_trial_dirs, load_run_metadata
    exportable = []
    total = 0
    failures = 0
    for trial_dir in iter_trial_dirs(jobs_dir, recursive=True):
        total += 1
        try:
            meta = load_run_metadata(trial_dir)
        except (ValueError, OSError) as exc:
            logger.warning("skip %s: invalid/missing run metadata: %s", trial_dir.name, exc)
            failures += 1
            continue
        if _agent_supports_atif(meta.get("agent_name", "")) and _has_trajectory(trial_dir):
            exportable.append(trial_dir)
        else:
            logger.info("skip %s: unsupported agent or missing main trajectory", trial_dir.name)
    return exportable, total, failures


def _to_single_dataset(result):
    """export_traces returns a Dataset (merged) or a dict of Datasets; normalize."""
    if isinstance(result, dict):
        from datasets import concatenate_datasets
        parts = [d for d in result.values() if d is not None and len(d) > 0]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    return result if (result is not None and len(result) > 0) else None


def _reward_of(row) -> float | None:
    """Parse the row's `result` field (reward string) into a float, or None."""
    val = row.get("result")
    try:
        reward = float(val)
        return reward if math.isfinite(reward) and not isinstance(val, bool) else None
    except (TypeError, ValueError):
        return None


def _mean_reward(ds) -> float | None:
    vals = [r for r in (_reward_of(row) for row in ds) if r is not None]
    return sum(vals) / len(vals) if vals else None


def _meets_reward(row, threshold: float) -> bool:
    reward = _reward_of(row)
    return reward is not None and reward >= threshold


def main(argv=None) -> int:
    args = _parse_args(argv)
    from harbor.models.trajectories.trajectory import Trajectory
    from harbor.utils.traces_utils import export_traces
    jobs_dir: Path = args.jobs_dir
    if not jobs_dir.exists():
        logger.error(f"jobs dir not found: {jobs_dir}")
        return 1
    if args.output.exists() and not args.overwrite and not args.dry_run:
        logger.error("output exists; use --overwrite explicitly: %s", args.output)
        return 1

    logger.info(f"scanning {jobs_dir} (filter={args.success_filter}, episodes={args.episodes})")

    # Pre-filter to ATIF-capable trials: export_traces raises (not skips) on
    # oracle/nop trials, and a mixed jobs/ dir usually contains them.
    exportable, n_total, failures = _discover_exportable_trials(jobs_dir)
    logger.info(f"found {n_total} trial(s); {len(exportable)} carry an exportable "
                f"trajectory (oracle/nop trials have none and are skipped)")
    if not exportable:
        logger.warning("no exportable trajectories — nothing to write")
        return 1

    # Export each qualifying trial and concatenate. Per-trial keeps one bad trial
    # from aborting the whole run.
    from datasets import concatenate_datasets
    parts = []
    for trial_dir in exportable:
        try:
            # Harbor can silently skip malformed JSON; validate before conversion.
            Trajectory.model_validate_json((trial_dir / "agent/trajectory.json").read_text())
            res = export_traces(
                trial_dir,
                recursive=False,
                episodes=args.episodes,
                success_filter=None if args.success_filter == "all" else args.success_filter,
                include_instruction=args.include_instruction,
                use_rich_progress=False,
                verbose=False,
                push=False,
                export_subagents=False,
            )
            part = _to_single_dataset(res)
            if part is None and args.success_filter == "all":
                raise ValueError("No exportable agent episodes in trajectory")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"skip {trial_dir.name}: export failed: {exc}")
            failures += 1
            continue
        if part is not None:
            parts.append(part)

    if failures and not args.allow_partial:
        logger.error("%d trial(s) failed; no output written. Fix inputs or use --allow-partial.", failures)
        return 2
    if failures:
        logger.warning("partial export explicitly enabled: %d failed trial(s)", failures)

    ds = concatenate_datasets(parts) if len(parts) > 1 else (parts[0] if parts else None)
    if ds is None:
        logger.warning("no rows produced after export — nothing to write")
        return 1

    n_rows = len(ds)
    mean_r = _mean_reward(ds)
    n_tasks = len({row.get("task") for row in ds})
    mean_str = f"{mean_r:.4f}" if mean_r is not None else "n/a"
    logger.info(f"discovered {n_rows} episode-rows across {n_tasks} task(s); mean reward={mean_str}")

    # Optional reward-threshold filter (filtered-SFT). Rows with unparseable reward are dropped.
    if args.min_reward is not None:
        before = n_rows
        ds = ds.filter(lambda row: _meets_reward(row, args.min_reward))
        logger.info(f"reward filter >= {args.min_reward}: kept {len(ds)}/{before} rows")
        n_rows = len(ds)

    if n_rows == 0:
        logger.warning("0 rows after filtering — nothing to write")
        return 1

    if args.dry_run:
        cols = sorted(ds.column_names)
        logger.info(f"[dry-run] would write {n_rows} rows -> {args.output}")
        logger.info(f"[dry-run] columns: {cols}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".jsonl", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        ds.to_json(str(tmp_path), force_ascii=False)
        if args.overwrite:
            os.replace(tmp_path, args.output)
        else:
            # Atomic no-clobber publication, even if another writer raced us.
            os.link(tmp_path, args.output)
    finally:
        tmp_path.unlink(missing_ok=True)
    final_mean = _mean_reward(ds)
    final_str = f"{final_mean:.4f}" if final_mean is not None else "n/a"
    logger.info(f"done: {n_rows} rows -> {args.output} (mean reward={final_str})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
