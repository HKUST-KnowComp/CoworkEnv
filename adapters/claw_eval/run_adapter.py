"""Convert explicitly selected, trusted Claw-Eval tasks into Harbor live tasks."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

try:
    from .adapter import ClawEvalAdapter, _contained_path, parse_task
except ImportError:
    from adapter import ClawEvalAdapter, _contained_path, parse_task


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--claw-eval-root', type=Path, default=os.getenv('CLAW_EVAL_ROOT'),
                        help='External Claw-Eval checkout (or CLAW_EVAL_ROOT).')
    parser.add_argument('--task-ids', nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/claw-eval'))
    parser.add_argument('--dry-run', action='store_true', help='Inspect source graders without writing tasks.')
    args = parser.parse_args()
    if args.claw_eval_root is None:
        parser.error('--claw-eval-root or CLAW_EVAL_ROOT is required')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        adapter = ClawEvalAdapter(args.output_dir, args.claw_eval_root)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    failures = 0
    for task_id in args.task_ids:
        try:
            task = _contained_path(adapter.source_root / 'tasks', task_id)
            if args.dry_run:
                parsed = parse_task(task)
                logging.info('%s: supported grader; payload/container execution not checked', parsed.source_id)
            else:
                logging.info('Generated %s', adapter.generate_task(task))
        except Exception as exc:
            logging.error('%s: %s', task_id, exc)
            failures += 1
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
