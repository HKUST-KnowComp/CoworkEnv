"""Regression checks for the released file-based tools (no environment runtime)."""

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from data_construction import d1_generator as generator
from data_construction import internal_dedup as dedup
from data_construction import leakage_scan as leakage
from data_construction import teacher_difficulty_filter as teacher


class GeneratorTests(unittest.TestCase):
    def spec(self, **overrides):
        args = dict(
            task_id="release_check", task_name="Release check", difficulty="easy",
            category="workflow", services=["contacts"], language="en",
            prompt_body="Find the requested contact.",
            fixtures={"contacts": {"contacts.json": {"contacts": []}}},
            completion_checks=[generator.CompletionCheck(
                name="lookup", weight=1.0, kind="tool_called",
                tool_name="contacts_search",
            )],
        )
        args.update(overrides)
        return generator.SeedSpec(**args)

    def test_generation_emits_valid_files_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = generator.generate(self.spec(), Path(tmp))
            task = yaml.safe_load((path / "task.yaml").read_text())
            self.assertEqual(task["task_id"], "release_check")
            self.assertNotIn("reference_solution", task)
            self.assertEqual(task["sandbox_files"], ["fixtures/contacts/contacts.json"])
            compile((path / "grader.py").read_text(), "grader.py", "exec")
            with self.assertRaises(FileExistsError):
                generator.generate(self.spec(), Path(tmp))

    def test_declared_identifier_leak_blocks_all_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self.spec(prompt_body="Use answer CONTACT-123.",
                             leakage_strings=["CONTACT-123"])
            with self.assertRaises(generator.LeakageError):
                generator.generate(spec, Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_rejects_unsafe_names_and_invalid_weights(self):
        for task_id in ["../outside", "bad/name", "bad\nname", "123"]:
            with self.subTest(task_id=task_id), self.assertRaises(generator.GenError):
                self.spec(task_id=task_id)
        with self.assertRaises(generator.GenError):
            self.spec(fixtures={"contacts": {"../outside.json": {}}})
        with self.assertRaises(generator.GenError):
            self.spec(fixtures={"other_service": {"state.json": {}}})
        for weight in [float("nan"), float("inf"), -1.0, 0.5]:
            with self.subTest(weight=weight), self.assertRaises(generator.GenError):
                self.spec(completion_checks=[generator.CompletionCheck(
                    name="invalid", weight=weight, kind="tool_called")])

    def test_catalog_tools_and_endpoints_match(self):
        total = 0
        for service in generator.SERVICE_REGISTRY:
            with self.subTest(service=service):
                catalog = generator._load_catalog(service)
                names = [tool["name"] for tool in catalog["tools"]]
                self.assertEqual(len(names), len(set(names)))
                self.assertEqual(set(names), {e["tool_name"] for e in catalog["endpoints"]})
                total += len(names)
        self.assertEqual(total, 44)


class LeakageTests(unittest.TestCase):
    def test_repeated_fixture_basenames_are_all_scanned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sub, text in [("one", "score 45.1"), ("two", "no answer")]:
                path = root / "fixtures" / sub / "state.txt"
                path.parent.mkdir(parents=True)
                path.write_text(text)
            leaks = leakage.check_task_dir(root, [45.1])
            self.assertEqual(len(leaks), 1)
            self.assertEqual(leaks[0].source, "fixture:fixtures/one/state.txt")

    def test_invalid_yaml_and_empty_directory_do_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                leakage.check_task_dir(root, [45.1])
            (root / "task.yaml").write_text("prompt: [unterminated")
            with self.assertRaises(yaml.YAMLError):
                leakage.check_task_dir(root, [45.1])

    def test_numeric_formatting_and_explicit_ignore(self):
        hits = leakage.scan_text("45.10% and 100", "prompt", [45.1, 100], {100})
        self.assertEqual([hit.gt_value for hit in hits], [45.1])


class ReportTests(unittest.TestCase):
    def test_dedup_reports_lexical_mode_and_identical_prompts(self):
        report = dedup.analyze({"one": "read the report and check totals",
                                "two": "read the report and check totals"})
        self.assertEqual(report["_meta"]["cosine_mode"], "tf_lexical")
        self.assertEqual(report["_meta"]["n_flagged_pairs"], 1)
        self.assertEqual(report["flagged_pairs"][0]["overlap_4gram"], 1.0)

    def test_teacher_collection_skips_invalid_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payloads = [
                {"verifier_result": {"rewards": {"completion": 0.25}}},
                {"completion": 0.75},
                {"completion": "NaN"}, {"completion": 2},
                {"verifier_result": {"rewards": None}}, [],
                {"completion": True},
            ]
            for i, payload in enumerate(payloads):
                path = root / f"task__trial{i}" / "result.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(payload))
            scores = teacher.collect(root)
            self.assertEqual(sorted(scores["task"]), [0.25, 0.75])
            summary = teacher.analyze(scores)["tasks"]["task"]
            self.assertEqual(summary["k"], 2)
            self.assertEqual(summary["mean_completion"], 0.5)
            self.assertEqual(summary["verdict"], "KEEP")

    def test_teacher_verdicts_retain_observed_score_semantics(self):
        report = teacher.analyze({"high": [1, 1], "low": [0, 0], "mid": [0.5, 0.5]})
        self.assertEqual(report["summary"], {"KEEP": 1, "TRIVIAL": 1, "UNSOLVABLE": 1})
        self.assertEqual(report["tasks"]["mid"]["variance"], 0.0)


if __name__ == "__main__":
    unittest.main()
