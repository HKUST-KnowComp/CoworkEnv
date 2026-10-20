"""Regression tests for training-data integrity and reward semantics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from training.sft.convert_data import convert_file, convert_record
from training.sft.validate_data import inspect_jsonl, template_messages, validate_messages
from training.rl.group_rewards import process_groups
from training.rl.prepare_prompts import collect_tasks


def atif_record():
    return {'reward': 0, 'trajectory': {'steps': [
        {'source': 'user', 'message': 'Read the item.'},
        {'source': 'agent', 'message': 'I will read it.', 'reasoning_content': 'Use the tool.',
         'tool_calls': [{'tool_call_id': 'call1', 'function_name': 'mcp__service__read', 'arguments': {'id': 'one'}}],
         'observation': {'results': [{'source_call_id': 'call1', 'content': 'real result'}]}},
        {'source': 'agent', 'message': 'Read complete.'},
    ]}}


class SFTDataTests(unittest.TestCase):
    def test_pairs_reasoning_and_zero_reward_are_preserved(self):
        row = convert_record(atif_record())
        self.assertEqual(row['reward'], 0)
        self.assertEqual([m['role'] for m in row['messages']], ['user', 'assistant', 'tool', 'assistant'])
        self.assertEqual(row['messages'][1]['reasoning_content'], 'Use the tool.')
        self.assertEqual(row['messages'][1]['tool_calls'][0]['function']['name'], 'mcp__service__read')
        self.assertEqual(row['messages'][2]['content'], 'real result')
        original = copy.deepcopy(row)
        rendered = template_messages(row['messages'])
        self.assertEqual(rendered[1]['tool_calls'][0]['function']['arguments'], {'id': 'one'})
        self.assertEqual(row, original)

    def test_missing_duplicate_and_unmatched_results_are_rejected(self):
        original = atif_record()
        for results in [[], [{'source_call_id': 'other', 'content': 'x'}],
                        [{'source_call_id': 'call1', 'content': 'x'}] * 2]:
            row = copy.deepcopy(original)
            row['trajectory']['steps'][1]['observation']['results'] = results
            with self.assertRaises(ValueError):
                convert_record(row)

    def test_multimodal_and_nonfinite_rewards_are_rejected(self):
        row = atif_record()
        row['trajectory']['steps'][0]['message'] = [{'type': 'image', 'url': 'not-loaded'}]
        with self.assertRaises(ValueError):
            convert_record(row)
        for reward in [float('nan'), True, 'RuntimeError']:
            row = atif_record()
            row['reward'] = reward
            with self.assertRaises(ValueError):
                convert_record(row)

    def test_harbor_conversations_and_missing_reward(self):
        row = convert_record({'conversations': [{'role': 'user', 'content': 'Request'}, {'role': 'assistant', 'content': 'Answer'}]})
        self.assertNotIn('reward', row)
        self.assertEqual(row['messages'][0]['role'], 'user')

    def test_long_tool_observations_are_not_truncated(self):
        row = atif_record()
        row['trajectory']['steps'][1]['observation']['results'][0]['content'] = 'z' * 5000
        self.assertEqual(len(convert_record(row)['messages'][2]['content']), 5000)

    def test_atomic_conversion_refuses_partial_output_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dest = Path(tmp) / 'input.jsonl', Path(tmp) / 'output.jsonl'
            src.write_text(json.dumps(atif_record()) + '\n{}\n')
            with self.assertRaises(ValueError):
                convert_file(src, dest)
            self.assertFalse(dest.exists())
            src.write_text(json.dumps(atif_record()) + '\n')
            self.assertEqual(convert_file(src, dest), 1)
            report = inspect_jsonl(dest)
            self.assertEqual(report['tool_call_count'], report['tool_result_count'])
            self.assertEqual(report['file'], 'output.jsonl')
            with self.assertRaises(FileExistsError):
                convert_file(src, dest)
            with self.assertRaises(ValueError):
                convert_file(src, src, overwrite=True)


class RewardTests(unittest.TestCase):
    def group(self, rewards):
        return [{'reward': value} for value in rewards]

    def test_grpo_floor_and_equal_rewards(self):
        equal = process_groups([self.group([.3, .3])], expected_size=2)
        self.assertEqual(equal['advantages'], [0, 0])
        self.assertEqual(equal['valid_mask'], [True, True])
        floored = process_groups([self.group([.49, .51])], expected_size=2, std_floor=.05)
        self.assertAlmostEqual(floored['advantages'][0], -.2)
        self.assertAlmostEqual(floored['advantages'][1], .2)

    def test_invalid_sample_rejects_whole_group(self):
        for value in [None, float('nan'), True]:
            result = process_groups([self.group([1, value])], expected_size=2)
            self.assertEqual(result['valid_mask'], [False, False])
            self.assertEqual(result['raw_rewards'], [None, None])
        group = self.group([1, 0])
        group[0]['infra_error'] = 'unavailable'
        self.assertFalse(process_groups([group], expected_size=2)['groups'][0]['keep'])

    def test_dictionary_reward_key_is_a_subkey(self):
        group = [{'reward': {'score': 0}}, {'reward': {'score': 1}}]
        self.assertEqual(process_groups([group], expected_size=2, reward_key='score')['advantages'], [-1, 1])
        self.assertFalse(process_groups([group], expected_size=2)['groups'][0]['keep'])

    def test_gdpo_requires_components_and_preserves_invalid_middle_slots(self):
        group = [{'reward': .2, 'reward_components': {'completion': 0, 'robustness': 1}},
                 {'reward': .8, 'reward_components': {'completion': 1, 'robustness': 0}}]
        result = process_groups([group, self.group([0, 1]), list(reversed(group))], method='gdpo', expected_size=2)
        self.assertEqual(result['valid_mask'], [True, True, False, False, True, True])
        self.assertEqual(result['advantages'], [-1, 1, 0, 0, 1, -1])

    def test_safety_multiplier_does_not_mean_zero_advantage(self):
        group = [{'reward': s, 'reward_components': {'completion': 1, 'robustness': 1, 'safety': s}} for s in [0, 1]]
        result = process_groups([group], method='gdpo', expected_size=2, safety_key='safety')
        self.assertEqual(result['advantages'], [-1, 1])
        self.assertEqual(process_groups([group], method='gdpo', expected_size=2)['advantages'], [0, 0])

    def test_configuration_errors_are_not_silent_fallbacks(self):
        for options in [{'method': 'ppo'}, {'expected_size': 0}, {'std_floor': float('nan')},
                        {'method': 'gdpo', 'reward_weights': {'completion': 2}}]:
            with self.assertRaises(ValueError):
                process_groups([self.group([0, 1])], **options)


class PromptTests(unittest.TestCase):
    def test_only_explicit_tasks_and_portable_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['T001', 'T002']:
                (root / name).mkdir()
                (root / name / 'task.yaml').write_text('task_id: ' + name)
            rows = collect_tasks(root, ['T002'])
            self.assertEqual([r['task_id'] for r in rows], ['T002'])
            self.assertEqual(rows[0]['task_path'], 'T002')
            for ids in [['../escape'], ['T002', 'T002']]:
                with self.assertRaises(ValueError):
                    collect_tasks(root, ids)


if __name__ == '__main__':
    unittest.main()
