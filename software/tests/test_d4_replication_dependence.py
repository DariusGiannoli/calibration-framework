import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from calibration_adequacy.config import load_task_bundle
from calibration_adequacy.criteria import evaluate_d1, evaluate_d4
from calibration_adequacy.models import CriterionStatus


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
D1_DEMO = REPOSITORY_ROOT / "software/examples/d1/synthetic_demo"
D4_DEMO = REPOSITORY_ROOT / "software/examples/d4/synthetic_demo"


class D4ReplicationDependenceTests(unittest.TestCase):
    def evaluate(self, dataset_path=None, task_path=None):
        dataset = dataset_path or D4_DEMO / "data.csv"
        task = task_path or D4_DEMO / "task.yaml"
        return evaluate_d4(dataset, load_task_bundle(task))

    def temporary_examples_copy(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        copied_examples = root / "examples"
        shutil.copytree(REPOSITORY_ROOT / "software/examples", copied_examples)
        return copied_examples

    def read_demo_rows(self):
        with (D4_DEMO / "data.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            return list(reader), list(reader.fieldnames or [])

    def write_rows(self, rows, fieldnames):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "data.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def modify_task(self, callback):
        copied_examples = self.temporary_examples_copy()
        task_path = copied_examples / "d4/synthetic_demo/task.yaml"
        task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        callback(task)
        task_path.write_text(
            yaml.safe_dump(task, sort_keys=False),
            encoding="utf-8",
        )
        return copied_examples, task_path

    def test_complete_independent_run_dataset_passes(self):
        result = self.evaluate()

        self.assertEqual(result.status, CriterionStatus.PASS)
        self.assertEqual(result.context["d1_status"], "PASS")
        self.assertEqual(result.metrics["observed_run_count"], 4)
        self.assertEqual(result.metrics["verified_independent_run_count"], 4)
        self.assertEqual(
            result.metrics["configuration_run_counts"],
            {"lissajous": 2, "spiral": 2},
        )
        self.assertGreater(
            result.metrics["minimum_effective_sample_size_observed"],
            12.0,
        )

    def test_d1_accepts_timestamp_restart_for_declared_runs(self):
        result = evaluate_d1(
            D4_DEMO / "data.csv",
            load_task_bundle(D4_DEMO / "task.yaml"),
        )

        self.assertEqual(result.status, CriterionStatus.PASS)
        self.assertEqual(result.metrics["rows_evaluated"], 32)

    def test_total_independent_run_requirement_can_fail(self):
        copied_examples, task_path = self.modify_task(
            lambda task: task["d4"].update(minimum_independent_runs=5)
        )

        result = self.evaluate(
            dataset_path=copied_examples / "d4/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "minimum_independent_runs_not_met",
            result.metrics["violations_by_code"],
        )

    def test_configuration_repetition_requirement_can_fail(self):
        def require_three_spirals(task):
            task["d4"]["minimum_runs_per_configuration"]["spiral"] = 3

        copied_examples, task_path = self.modify_task(require_three_spirals)
        result = self.evaluate(
            dataset_path=copied_examples / "d4/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "minimum_configuration_runs_not_met",
            result.metrics["violations_by_code"],
        )

    def test_effective_sample_size_requirement_can_fail(self):
        copied_examples, task_path = self.modify_task(
            lambda task: task["d4"].update(
                minimum_effective_sample_size=14.0
            )
        )

        result = self.evaluate(
            dataset_path=copied_examples / "d4/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "minimum_effective_sample_size_not_met",
            result.metrics["violations_by_code"],
        )
        self.assertEqual(result.metrics["limiting_signal"], "sensor.c1")

    def test_duplicate_acquisition_cannot_create_independent_runs(self):
        def duplicate_acquisition(task):
            task["d4"]["run_evidence"][1]["acquisition_id"] = "acquisition_01"

        copied_examples, task_path = self.modify_task(duplicate_acquisition)
        result = self.evaluate(
            dataset_path=copied_examples / "d4/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "duplicate_acquisition_id",
            result.metrics["violations_by_code"],
        )

    def test_segment_not_declared_separate_cannot_count_as_run(self):
        def mark_not_separate(task):
            task["d4"]["run_evidence"][1]["separate_acquisition"] = False

        copied_examples, task_path = self.modify_task(mark_not_separate)
        result = self.evaluate(
            dataset_path=copied_examples / "d4/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "separate_acquisition_not_established",
            result.metrics["violations_by_code"],
        )

    def test_missing_run_evidence_is_indeterminate_not_a_false_failure(self):
        def remove_one_run_evidence(task):
            task["d4"]["run_evidence"] = task["d4"]["run_evidence"][:-1]

        copied_examples, task_path = self.modify_task(remove_one_run_evidence)
        result = self.evaluate(
            dataset_path=copied_examples / "d4/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.INDETERMINATE)
        self.assertIn(
            "task.d4.run_evidence.lissajous_02",
            result.missing_evidence,
        )
        self.assertEqual(result.metrics["maximum_possible_independent_run_count"], 4)

    def test_missing_lag_declaration_is_indeterminate(self):
        def remove_lag(task):
            task["d4"]["signals"][0]["maximum_autocorrelation_lag"] = None

        copied_examples, task_path = self.modify_task(remove_lag)
        result = self.evaluate(
            dataset_path=copied_examples / "d4/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.INDETERMINATE)
        self.assertIn(
            "task.d4.signals.sensor.c1.maximum_autocorrelation_lag",
            result.missing_evidence,
        )

    def test_constant_relevant_signal_cannot_support_autocorrelation(self):
        rows, fieldnames = self.read_demo_rows()
        for row in rows:
            row["c1"] = "0.0"
        dataset = self.write_rows(rows, fieldnames)

        result = self.evaluate(dataset_path=dataset)

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "autocorrelation_undefined_constant_signal",
            result.metrics["violations_by_code"],
        )

    def test_d1_failure_blocks_d4(self):
        rows, fieldnames = self.read_demo_rows()
        rows[1]["reference_time_s"] = "0.100"
        dataset = self.write_rows(rows, fieldnames)

        result = self.evaluate(dataset_path=dataset)

        self.assertEqual(result.status, CriterionStatus.INDETERMINATE)
        self.assertEqual(result.context["d1_status"], "FAIL")
        self.assertIn("prerequisite.D1_PASS", result.missing_evidence)

    def test_missing_d4_configuration_is_indeterminate(self):
        result = self.evaluate(
            dataset_path=D1_DEMO / "data.csv",
            task_path=D1_DEMO / "task.yaml",
        )

        self.assertEqual(result.status, CriterionStatus.INDETERMINATE)
        self.assertIn("task.d4", result.missing_evidence)


if __name__ == "__main__":
    unittest.main()
