"""Convert selected trusted ClawMark filesystem/email tasks to Harbor."""
from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path

try:
    from .adapter import ClawMarkAdapter, contained_path, offline_blockers, parse_task
except ImportError:
    from adapter import ClawMarkAdapter, contained_path, offline_blockers, parse_task


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clawmark-root', type=Path, default=os.getenv('CLAWMARK_ROOT'))
    p.add_argument('--task-ids', nargs='+', required=True, help='Explicit domain/task IDs, e.g. journalist/task5')
    p.add_argument('--output-dir', type=Path, default=Path('outputs/clawmark'))
    p.add_argument('--reward-strategy', choices=['mean', 'final'], default='mean')
    p.add_argument('--oracle-root', type=Path, help='Optional trusted scripts: <source_id>/<stage>.sh')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.clawmark_root is None or not args.clawmark_root.is_dir():
        p.error('Supply --clawmark-root or CLAWMARK_ROOT pointing to a trusted checkout')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    adapter = ClawMarkAdapter(args.output_dir, reward_strategy=args.reward_strategy, oracle_root=args.oracle_root)
    failures = 0
    for task_id in args.task_ids:
        try:
            task = contained_path(args.clawmark_root / 'tasks', task_id)
            if args.dry_run:
                parsed = parse_task(task)
                blockers = offline_blockers(parsed)
                if blockers:
                    raise ValueError('; '.join(blockers))
                logging.info('%s: supported source structure; stages=%s; containers not executed', task_id, parsed.stage_ids)
            else:
                logging.info('Generated %s', adapter.generate_task(task))
        except Exception as exc:
            logging.error('%s: %s', task_id, exc)
            failures += 1
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
