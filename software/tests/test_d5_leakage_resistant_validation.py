import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from calibration_adequacy.config import load_task_bundle
from calibration_adequacy.criteria import evaluate_d5
from calibration_adequacy.models import CriterionStatus


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
D1_DEMO = REPOSITORY_ROOT / "software/examples/d1/synthetic_demo"
D5_DEMO = REPOSITORY_ROOT / "software/examples/d5/synthetic_demo"


class D5LeakageResistantValidationTests(unittest.TestCase):
    def evaluate(self, dataset_path=None, task_path=None):
        dataset = dataset_path or D5_DEMO / "data.csv"
        task = task_path or D5_DEMO / "task.yaml"
        return evaluate_d5(dataset, load_task_bundle(task))

    def temporary_examples_copy(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        copied_examples = root / "examples"
        shutil.copytree(REPOSITORY_ROOT / "software/examples", copied_examples)
        return copied_examples

    def modify_task(self, callback):
        copied_examples = self.temporary_examples_copy()
        task_path = copied_examples / "d5/synthetic_demo/task.yaml"
        task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        callback(task)
        task_path.write_text(
            yaml.safe_dump(task, sort_keys=False),
            encoding="utf-8",
        )
        return copied_examples, task_path

    def read_demo_rows(self):
        with (D5_DEMO / "data.csv").open(
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

    def test_complete_frozen_run_level_split_passes(self):
        result = self.evaluate()

        self.assertEqual(result.status, CriterionStatus.PASS)
        self.assertEqual(result.context["d1_status"], "PASS")
        self.assertEqual(result.metrics["development_unit_count"], 3)
        self.assertEqual(result.metrics["test_unit_count"], 1)
        self.assertEqual(result.metrics["development_sample_count"], 24)
        self.assertEqual(result.metrics["test_sample_count"], 8)
        self.assertEqual(result.metrics["development_design_rank"], 4)
        for rmse in result.metrics["test_rmse"].values():
            self.assertAlmostEqual(rmse, 0.0, places=12)

    def test_development_and_test_run_overlap_fails(self):
        def create_overlap(task):
            task["d5"]["development_units"].append("test_01")

        copied_examples, task_path = self.modify_task(create_overlap)
        result = self.evaluate(
            dataset_path=copied_examples / "d5/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "development_test_unit_overlap",
            result.metrics["violations_by_code"],
        )

    def test_every_observed_run_must_be_partitioned(self):
        def omit_test_run(task):
            task["d5"]["test_units"] = []

        copied_examples, task_path = self.modify_task(omit_test_run)
        result = self.evaluate(
            dataset_path=copied_examples / "d5/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "observed_unit_not_partitioned",
            result.metrics["violations_by_code"],
        )
        self.assertEqual(result.metrics["unassigned_observed_units"], ["test_01"])

    def test_declared_run_absent_from_dataset_fails(self):
        def add_absent_test_run(task):
            task["d5"]["test_units"].append("test_02")

        copied_examples, task_path = self.modify_task(add_absent_test_run)
        result = self.evaluate(
            dataset_path=copied_examples / "d5/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "declared_unit_not_observed",
            result.metrics["violations_by_code"],
        )

    def test_minimum_held_out_run_requirement_can_fail(self):
        copied_examples, task_path = self.modify_task(
            lambda task: task["d5"].update(minimum_test_units=2)
        )
        result = self.evaluate(
            dataset_path=copied_examples / "d5/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "minimum_test_units_not_met",
            result.metrics["violations_by_code"],
        )

    def test_test_data_used_for_preprocessing_fails(self):
        def contaminate_preprocessing(task):
            task["d5"]["data_use"][
                "data_dependent_preprocessing"
            ] = "includes_test"

        copied_examples, task_path = self.modify_task(contaminate_preprocessing)
        result = self.evaluate(
            dataset_path=copied_examples / "d5/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "test_data_used_during_development",
            result.metrics["violations_by_code"],
        )

    def test_post_hoc_split_fails(self):
        copied_examples, task_path = self.modify_task(
            lambda task: task["d5"].update(
                split_frozen_before_development=False
            )
        )
        result = self.evaluate(
            dataset_path=copied_examples / "d5/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "split_not_frozen_before_development",
            result.metrics["violations_by_code"],
        )

    def test_test_feedback_into_further_development_fails(self):
        def mark_test_feedback(task):
            task["d5"]["data_use"][
                "test_results_used_for_further_development"
            ] = True

        copied_examples, task_path = self.modify_task(mark_test_feedback)
        result = self.evaluate(
            dataset_path=copied_examples / "d5/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "test_results_fed_back_into_development",
            result.metrics["violations_by_code"],
        )

    def test_missing_provenance_declaration_is_indeterminate(self):
        def remove_model_selection_scope(task):
            task["d5"]["data_use"]["model_selection"] = None

        copied_examples, task_path = self.modify_task(
            remove_model_selection_scope
        )
        result = self.evaluate(
            dataset_path=copied_examples / "d5/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.INDETERMINATE)
        self.assertIn(
            "task.d5.data_use.model_selection",
            result.missing_evidence,
        )

    def test_d1_failure_blocks_d5(self):
        rows, fieldnames = self.read_demo_rows()
        rows[1]["reference_time_s"] = "0.100"
        dataset = self.write_rows(rows, fieldnames)

        result = self.evaluate(dataset_path=dataset)

        self.assertEqual(result.status, CriterionStatus.INDETERMINATE)
        self.assertEqual(result.context["d1_status"], "FAIL")
        self.assertIn("prerequisite.D1_PASS", result.missing_evidence)

    def test_missing_d5_configuration_is_indeterminate(self):
        result = self.evaluate(
            dataset_path=D1_DEMO / "data.csv",
            task_path=D1_DEMO / "task.yaml",
        )

        self.assertEqual(result.status, CriterionStatus.INDETERMINATE)
        self.assertIn("task.d5", result.missing_evidence)

    def test_high_test_error_does_not_change_d5_protocol_status(self):
        rows, fieldnames = self.read_demo_rows()
        for row in rows:
            if row["run_id"] == "test_01":
                for channel in ("Fx_ref", "Fy_ref", "Fz_ref"):
                    row[channel] = str(float(row[channel]) + 5.0)
        dataset = self.write_rows(rows, fieldnames)

        result = self.evaluate(dataset_path=dataset)

        self.assertEqual(result.status, CriterionStatus.PASS)
        for rmse in result.metrics["test_rmse"].values():
            self.assertAlmostEqual(rmse, 5.0, places=12)
        sensitivity = result.metrics["estimated_sensitivity_matrix"]
        for row_index, row in enumerate(sensitivity):
            for column_index, value in enumerate(row):
                expected = 1.0 if row_index == column_index else 0.0
                self.assertAlmostEqual(value, expected, places=12)
        self.assertFalse(result.metrics["performance_threshold_applied"])

    def test_model_and_reference_dimension_mismatch_fails_cleanly(self):
        def change_output_dimension(task):
            task["d3"]["output_dimension"] = 2

        copied_examples, task_path = self.modify_task(change_output_dimension)
        result = self.evaluate(
            dataset_path=copied_examples / "d5/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "model_output_dimension_mismatch",
            result.metrics["violations_by_code"],
        )


if __name__ == "__main__":
    unittest.main()
