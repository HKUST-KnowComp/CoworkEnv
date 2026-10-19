"""ClawMark release checks. Small synthetic fixtures are unit tests, not tasks.

Set CLAWMARK_ROOT to a trusted upstream checkout for real-source conversion.
No test runs task seeding on the host or launches a model/container.
"""
import asyncio
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from adapters.clawmark import adapter

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'adapters/clawmark/runtime'


def load_runner(filename):
    # Generated runners import the sibling ctx_runtime module inside a container.
    with patch.object(sys, 'path', [str(RUNTIME), *sys.path]):
        spec = importlib.util.spec_from_file_location(
            '_clawmark_test_' + filename, adapter.TEMPLATE_DIR / (filename + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class ConversionTests(unittest.TestCase):
    def fixture(self, root):
        task = root / 'source/domain/task'
        task.mkdir(parents=True)
        (task / 'task.py').write_text('''
METADATA = {'id': 'unit_stage', 'environments': ['filesystem']}
PROMPT = 'Preserve the document.'
async def checked(ctx):
    return True
async def stage0(ctx):
    raise RuntimeError('Seeding must never run during conversion')
    return {'notification': 'Start the task.'}
RUBRIC = {'stage0': [{'id': 'document', 'weight': 1, 'checker': checked}]}
''')
        (task / 'assets').mkdir()
        (task / 'assets/input.txt').write_text('initial')
        (task / 'inject/stage0').mkdir(parents=True)
        (task / 'inject/stage0/update.txt').write_text('updated')
        return task

    def test_conversion_retains_resources_without_executing_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self.fixture(root)
            out = adapter.ClawMarkAdapter(root / 'output').generate_task(task)
            self.assertIn('Start the task.', (out / 'steps/stage0/instruction.md').read_text())
            for payload in ['tests', 'workdir/.coworkenv-seed']:
                base = out / 'steps/stage0' / payload
                self.assertEqual((base / 'inject/stage0/update.txt').read_text(), 'updated')
                self.assertEqual((base / 'clawmark_task.py').read_bytes(), (task / 'task.py').read_bytes())
                self.assertTrue((base / 'clawmark_vendor/LICENSE').is_file())
            self.assertFalse(list(out.rglob('solve.sh')))
            self.assertIn("trap 'rm -rf -- .coworkenv-seed' EXIT", (out / 'steps/stage0/workdir/setup.sh').read_text())
            for script in out.rglob('*.sh'):
                subprocess.run(['bash', '-n', str(script)], check=True)
            with self.assertRaises(FileExistsError):
                adapter.ClawMarkAdapter(root / 'output').generate_task(task)

    def test_dynamic_notifications_rejected(self):
        tree = ast.parse("async def stage0(ctx):\n    return {'notification': ctx.message}\n")
        with self.assertRaises(ValueError):
            adapter._static_notification(tree, 'stage0')

    def test_rejects_unsupported_backend_and_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self.fixture(root)
            path = task / 'task.py'
            path.write_text(path.read_text().replace("['filesystem']", "['calendar']"))
            with self.assertRaisesRegex(ValueError, 'unsupported runtime'):
                adapter.ClawMarkAdapter(root / 'output').generate_task(task)
            self.assertFalse((root / 'output').exists())
            for unsafe in ['../outside', '/absolute']:
                with self.assertRaises(ValueError):
                    adapter.contained_path(root, unsafe)
            (root / 'linked').symlink_to(root.parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                adapter.contained_path(root, 'linked')

    def test_rejects_symlink_assets_without_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self.fixture(root)
            (task / 'assets/linked').symlink_to(task / 'task.py')
            with self.assertRaises(ValueError):
                adapter.ClawMarkAdapter(root / 'output').generate_task(task)
            self.assertEqual(list((root / 'output').iterdir()), [])


class RuntimeTests(unittest.TestCase):
    def test_seeding_shell_failure_propagates(self):
        setup = load_runner('stage_setup')
        async def stage(ctx):
            await ctx.sandbox.exec('a failing command')
        ctx = SimpleNamespace(sandbox=SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=1))))
        module = SimpleNamespace(METADATA={}, stage0=stage)
        with patch.object(setup, '_load_task_module', return_value=module), patch.object(setup, 'build_ctx', AsyncMock(return_value=ctx)):
            with self.assertRaisesRegex(RuntimeError, 'shell command failed'):
                asyncio.run(setup._seed('stage0'))

    def test_seeding_email_failure_propagates(self):
        setup = load_runner('stage_setup')
        async def stage(ctx):
            await ctx.email.send_email('sender', 'recipient', 'subject', 'body')
        ctx = SimpleNamespace(sandbox=SimpleNamespace(exec=AsyncMock()), email=SimpleNamespace(send_email=AsyncMock(return_value=False)))
        module = SimpleNamespace(METADATA={}, stage0=stage)
        with patch.object(setup, '_load_task_module', return_value=module), patch.object(setup, 'build_ctx', AsyncMock(return_value=ctx)):
            with self.assertRaisesRegex(RuntimeError, 'email delivery failed'):
                asyncio.run(setup._seed('stage0'))

    def test_weighted_scoring_and_checker_failure(self):
        runner = load_runner('test_runner')
        entries = [
            {'id': 'pass', 'weight': 2, 'checker': AsyncMock(return_value=True)},
            {'id': 'fail', 'weight': 1, 'checker': AsyncMock(return_value=False)},
            {'id': 'error', 'weight': 1, 'checker': AsyncMock(side_effect=ValueError('bad result'))},
        ]
        with patch.object(runner, 'build_ctx', AsyncMock(return_value=object())):
            score, results = asyncio.run(runner._run(entries, SimpleNamespace(METADATA={})))
            self.assertEqual(score, .5)
            self.assertFalse(results[2][1])

    def test_bad_runtime_removes_stale_reward_and_returns_failure(self):
        runner = load_runner('test_runner')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'reward.json').write_text('{"reward": 1}')
            with patch.object(runner, 'VERIFIER_DIR', path), patch.object(runner, '_load_task_module', side_effect=ValueError('invalid task')), patch.object(sys, 'argv', ['test_runner.py', 'stage0']):
                self.assertEqual(runner.main(), 1)
            self.assertFalse((path / 'reward.json').exists())
            self.assertEqual(json.loads((path / 'error.json').read_text())['stage'], 'stage0')

    def test_local_copy_handles_literal_shell_metacharacters(self):
        load_runner('stage_setup')
        from clawmark_vendor.sandbox.local import LocalSandbox
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / 'source'
            src.mkdir()
            (src / 'payload.txt').write_text('payload')
            target = root / 'directory ; $literal'
            asyncio.run(LocalSandbox().upload_dir(src, str(target)))
            self.assertEqual((target / 'payload.txt').read_text(), 'payload')

    def test_runtime_has_no_external_python_imports(self):
        allowed = sys.stdlib_module_names | {'clawmark_vendor'}
        for path in RUNTIME.rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = [x.name.split('.')[0] for x in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [(node.module or '').split('.')[0]]
                else:
                    continue
                self.assertTrue(set(names) <= allowed, f'{path.name}: {names}')


@unittest.skipUnless(os.getenv('CLAWMARK_ROOT') and importlib.util.find_spec('harbor'), 'Set CLAWMARK_ROOT and install requirements-export.txt')
class RealSourceTests(unittest.TestCase):
    def test_real_task_conversion_and_harbor_schema(self):
        import tomllib
        from harbor.models.task.config import TaskConfig
        source = Path(os.environ['CLAWMARK_ROOT'])
        for task_id in ['eda/task1', 'journalist/task5', 'journalist/task7']:
            with self.subTest(task=task_id), tempfile.TemporaryDirectory() as tmp:
                task = source / 'tasks' / task_id
                out = adapter.ClawMarkAdapter(Path(tmp)).generate_task(task)
                config = TaskConfig.model_validate(tomllib.loads((out / 'task.toml').read_text()))
                self.assertGreaterEqual(len(config.steps), 2)
                self.assertTrue((out / 'environment/docker-compose.yaml').exists())
                for step in (out / 'steps').iterdir():
                    self.assertEqual((step / 'tests/clawmark_task.py').read_bytes(), (task / 'task.py').read_bytes())
                    for resource in (task / 'inject').rglob('*'):
                        if resource.is_file():
                            dest = step / 'tests/inject' / resource.relative_to(task / 'inject')
                            self.assertEqual(dest.read_bytes(), resource.read_bytes())
                    for script in step.rglob('*.sh'):
                        subprocess.run(['bash', '-n', str(script)], check=True)


if __name__ == '__main__':
    unittest.main()
