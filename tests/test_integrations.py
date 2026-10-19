"""Release regression checks; fixtures here are tests, not benchmark data."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import export_rollouts as export

ROOT = Path(__file__).resolve().parents[1]
HAS_HARBOR = importlib.util.find_spec('harbor') is not None
HAS_PYDANTIC = importlib.util.find_spec('pydantic') is not None
HAS_BRIDGE = importlib.util.find_spec('fastmcp') is not None


class RewardFilterTests(unittest.TestCase):
    def test_zero_is_a_valid_reward(self):
        self.assertTrue(export._meets_reward({'result': '0.0'}, 0))
        self.assertFalse(export._meets_reward({'result': '-0.1'}, 0))
        for value in ['nan', 'inf', '-inf', 'RuntimeError', None, True]:
            self.assertIsNone(export._reward_of({'result': value}))

    def test_only_main_trajectory_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp)
            (trial / 'agent').mkdir()
            (trial / 'agent/trajectory.other.json').write_text('{}')
            self.assertFalse(export._has_trajectory(trial))


@unittest.skipUnless(HAS_HARBOR, 'Install requirements-export.txt')
class HarborExportTests(unittest.TestCase):
    def make_trial(self, root, name, reward=0):
        trial = root / name
        (trial / 'agent').mkdir(parents=True)
        (trial / 'result.json').write_text(json.dumps({
            'task_name': 'export-regression', 'trial_name': name,
            'config': {'agent': {'name': 'claude-code', 'model_name': 'test-model'}},
            'verifier_result': {'rewards': {'reward': reward}},
        }))
        (trial / 'agent/trajectory.json').write_text(json.dumps({
            'schema_version': 'ATIF-v1.8',
            'agent': {'name': 'claude-code', 'version': 'test', 'model_name': 'test-model'},
            'steps': [
                {'step_id': 1, 'source': 'user', 'message': 'Test instruction'},
                {'step_id': 2, 'source': 'agent', 'message': 'Test response'},
            ],
        }))
        return trial

    def test_real_harbor_export_preserves_zero_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_trial(root / 'jobs', 'zero')
            out = root / 'output.jsonl'
            args = ['--jobs-dir', str(root / 'jobs'), '--output', str(out), '--min-reward', '0']
            self.assertEqual(export.main(args), 0)
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]['result']), 0)
            self.assertEqual(rows[0]['conversations'][-1]['content'], 'Test response')
            before = out.read_bytes()
            self.assertEqual(export.main(args), 1)
            self.assertEqual(out.read_bytes(), before)

    def test_corrupt_trajectory_cannot_silently_produce_partial_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_trial(root / 'jobs', 'good', 1)
            bad = self.make_trial(root / 'jobs', 'bad')
            (bad / 'agent/trajectory.json').write_text('{')
            out = root / 'output.jsonl'
            args = ['--jobs-dir', str(root / 'jobs'), '--output', str(out)]
            self.assertEqual(export.main(args), 2)
            self.assertFalse(out.exists())
            self.assertEqual(export.main(args + ['--allow-partial']), 0)
            self.assertEqual(len(out.read_text().splitlines()), 1)


@unittest.skipUnless(HAS_PYDANTIC, 'Install requirements-adapters.txt')
class VerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from adapters.claw_eval.template.tests import test_runner
        cls.runner = test_runner

    def test_uses_last_agent_answer_and_rejects_missing_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trajectory.json'
            path.write_text(json.dumps({'steps': [
                {'source': 'user', 'message': 'user text'},
                {'source': 'agent', 'message': 'real answer'},
                {'source': 'agent', 'message': 'Executed tool', 'tool_calls': [{}]},
            ]}))
            self.assertEqual(self.runner._final_assistant_text(path), 'real answer')
            path.write_text('{"steps": []}')
            with self.assertRaises(ValueError):
                self.runner._final_assistant_text(path)

    def test_missing_evidence_removes_stale_reward(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'reward.json').write_text('{"reward": 1}')
            with patch.object(self.runner, 'VERIFIER_DIR', root), patch.object(self.runner, 'LIVE_CONFIG_PATH', root / 'missing.json'):
                self.assertEqual(self.runner.main(), 1)
            self.assertFalse((root / 'reward.json').exists())
            self.assertEqual(json.loads((root / 'error.json').read_text())['status'], 'invalid_evidence')

    def test_source_paths_cannot_escape_checkout(self):
        from adapters.claw_eval.adapter import _contained_path
        with tempfile.TemporaryDirectory() as tmp:
            for path in ['../outside', '/absolute']:
                with self.assertRaises(ValueError):
                    _contained_path(Path(tmp), path)


@unittest.skipUnless(HAS_BRIDGE, 'Install bridge requirements.txt')
class BridgeTests(unittest.TestCase):
    def test_mcp_schema_and_dispatch_status(self):
        from adapters.claw_eval.runtime.mcp_bridge import bridge_server as bridge
        from fastmcp import Client
        import httpx

        async def check():
            config = {'task_id': 'unit', 'tools': [{
                'name': 'read_item', 'input_schema': {
                    'type': 'object', 'properties': {'id': {'type': 'string', 'enum': ['missing']}},
                    'required': ['id'], 'additionalProperties': False,
                }, 'url': 'http://localhost:9109/items',
            }]}
            transport = httpx.MockTransport(lambda req: httpx.Response(404, json={'error': 'not found'}))
            original = httpx.AsyncClient
            with patch.object(bridge, '_load_config', return_value=config):
                server = bridge.build_server()
            with patch.object(bridge.httpx, 'AsyncClient', side_effect=lambda **kwargs: original(transport=transport, **kwargs)):
                async with Client(server) as client:
                    tools = await client.list_tools()
                    self.assertEqual(tools[0].inputSchema, config['tools'][0]['input_schema'])
                    result = await client.call_tool('read_item', {'id': 'missing'})
                    self.assertEqual(json.loads(result.content[0].text)['status_code'], 404)
            # Query the actual read-only evidence route via ASGI, without a server socket.
            async with original(transport=httpx.ASGITransport(app=server.http_app()), base_url='http://bridge') as client:
                evidence = (await client.get('/dispatches')).json()
                self.assertEqual(evidence['task_id'], 'unit')
                self.assertEqual(evidence['dispatches'][0]['response_status'], 404)
                self.assertEqual((await client.post('/dispatches', json={})).status_code, 405)
        asyncio.run(check())


@unittest.skipUnless(HAS_PYDANTIC and os.getenv('CLAW_EVAL_ROOT'), 'Set CLAW_EVAL_ROOT to a trusted upstream checkout')
class UpstreamConversionTests(unittest.TestCase):
    def test_three_real_tasks_generate_without_solution_or_self_trace(self):
        from adapters.claw_eval.adapter import ClawEvalAdapter
        source = Path(os.environ['CLAW_EVAL_ROOT'])
        with tempfile.TemporaryDirectory() as tmp:
            adapter = ClawEvalAdapter(Path(tmp), source)
            for task in ['T087_pinbench_market_news_brief', 'T090_pinbench_config_change_plan', 'T095_pinbench_apm_market_research']:
                with self.subTest(task=task):
                    out = adapter.generate_task(source / 'tasks' / task)
                    self.assertTrue((out / 'environment/mcp-bridge/requirements.txt').is_file())
                    self.assertFalse((out / 'solution').exists())
                    config = json.loads((out / 'tests/live_config.json').read_text())
                    self.assertEqual(config['task_id'], task)
                    if HAS_HARBOR:
                        import tomllib
                        from harbor.models.task.config import TaskConfig
                        TaskConfig.model_validate(tomllib.loads((out / 'task.toml').read_text()))
                    with self.assertRaises(FileExistsError):
                        adapter.generate_task(source / 'tasks' / task)


if __name__ == '__main__':
    unittest.main()
