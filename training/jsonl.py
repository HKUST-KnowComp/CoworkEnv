"""Small atomic JSONL helpers shared by the released training tools."""
import json
import os
from pathlib import Path
import tempfile


def read_records(path):
    with Path(path).open(encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError('expected an object')
            except (ValueError, TypeError) as exc:
                raise ValueError(f'line {line_number}: invalid JSON object') from exc
            yield record


def write_records(records, output, *, overwrite=False):
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError('Output exists; choose a fresh path or pass --overwrite')
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.jsonl-', dir=output.parent)
    count = 0
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                count += 1
        if not count:
            raise ValueError('No records to write')
        if overwrite:
            os.replace(name, output)
        else:
            # Atomic publication which also refuses concurrent overwrites.
            os.link(name, output)
        return count
    finally:
        Path(name).unlink(missing_ok=True)
