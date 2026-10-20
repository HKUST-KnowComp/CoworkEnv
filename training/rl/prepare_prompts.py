"""Build an explicit Claw-Eval task manifest for a custom RL rollout adapter.

Adapted from train/rl/prepare_prompts.py. No hardcoded paths, hidden exclusions,
automatic train/test split or claim that every selected task is executable.
"""
import argparse
from pathlib import Path
import yaml
from training.jsonl import write_records


def collect_tasks(tasks_root, task_ids, *, absolute_paths=False):
    root = Path(tasks_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError('tasks_root must be an existing directory')
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError('Supply a nonempty selection of unique task IDs')
    rows = []
    for task_id in task_ids:
        rel = Path(task_id)
        if rel.is_absolute() or '..' in rel.parts or str(rel) in {'.', ''}:
            raise ValueError('Task IDs must be relative to tasks_root')
        task = (root / rel).resolve()
        config = (task / 'task.yaml').resolve()
        if not task.is_relative_to(root) or not config.is_relative_to(root):
            raise ValueError('Task path escapes tasks_root')
        if not isinstance(yaml.safe_load(config.read_text()), dict):
            raise ValueError('task.yaml must contain a mapping')
        rows.append({'prompt': [{'role': 'user', 'content': task_id}],
                     'task_id': task_id,
                     'task_path': str(task) if absolute_paths else task.relative_to(root).as_posix()})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks-root', type=Path, required=True)
    p.add_argument('--task-ids', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--absolute-paths', action='store_true', help='Resolve paths for a specific deployment; do not publish that manifest blindly')
    a = p.parse_args()
    rows = collect_tasks(a.tasks_root, a.task_ids, absolute_paths=a.absolute_paths)
    print('Selected tasks:', write_records(rows, a.output))


if __name__ == '__main__':
    main()
