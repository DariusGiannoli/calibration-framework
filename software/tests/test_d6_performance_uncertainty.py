import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from calibration_adequacy.config import load_task_bundle
from calibration_adequacy.criteria import evaluate_d6
from calibration_adequacy.models import CriterionStatus


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
D5_DEMO = REPOSITORY_ROOT / "software/examples/d5/synthetic_demo"
D6_DEMO = REPOSITORY_ROOT / "software/examples/d6/synthetic_demo"


class D6PerformanceUncertaintyTests(unittest.TestCase):
    def evaluate(self, dataset_path=None, task_path=None):
        dataset = dataset_path or D6_DEMO / "data.csv"
        task = task_path or D6_DEMO / "task.yaml"
        return evaluate_d6(dataset, load_task_bundle(task))

    def temporary_examples_copy(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        copied_examples = root / "examples"
        shutil.copytree(REPOSITORY_ROOT / "software/examples", copied_examples)
        return copied_examples

    def modify_task(self, callback):
        copied_examples = self.temporary_examples_copy()
        task_path = copied_examples / "d6/synthetic_demo/task.yaml"
        task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        callback(task)
        task_path.write_text(
            yaml.safe_dump(task, sort_keys=False),
            encoding="utf-8",
        )
        return copied_examples, task_path

    def read_demo_rows(self):
        with (D6_DEMO / "data.csv").open(
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

    def test_complete_run_bootstrap_evidence_passes(self):
        result = self.evaluate()

        self.assertEqual(result.status, CriterionStatus.PASS)
        self.assertEqual(result.context["d5_status"], "PASS")
        self.assertEqual(result.metrics["dataset_adequacy_status"], "PASS")
        self.assertEqual(
            result.metrics["calibration_acceptance_status"],
            "PASS",
        )
        self.assertEqual(result.metrics["bootstrap_unit"], "run")
        self.assertEqual(result.metrics["test_unit_count"], 4)
        self.assertEqual(result.metrics["test_sample_count"], 32)
        self.assertEqual(set(result.metrics["regions"]), {
            "negative_Fx",
            "nonnegative_Fx",
        })
        for axis in result.metrics["axes"].values():
            self.assertTrue(axis["performance_evidence_precise"])
            self.assertTrue(axis["calibration_axis_accepted"])

    def test_interval_wider_than_declared_precision_fails_adequacy(self):
        def tighten_precision(task):
            task["d6"]["axes"]["Fx"]["maximum_interval_half_width"] = 0.001

        copied_examples, task_path = self.modify_task(tighten_precision)
        result = self.evaluate(
            dataset_path=copied_examples / "d6/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertEqual(result.metrics["dataset_adequacy_status"], "FAIL")
        self.assertIn(
            "metric_interval_too_wide",
            result.metrics["violations_by_code"],
        )

    def test_precise_failure_is_not_mislabeled_as_inadequate_data(self):
        rows, fieldnames = self.read_demo_rows()
        for row in rows:
            if row["run_id"].startswith("test_"):
                for channel in ("c1", "c2", "c3"):
                    row[channel] = str(float(row[channel]) + 1.0)
        dataset = self.write_rows(rows, fieldnames)

        result = self.evaluate(dataset_path=dataset)

        self.assertEqual(result.status, CriterionStatus.PASS)
        self.assertEqual(result.metrics["dataset_adequacy_status"], "PASS")
        self.assertEqual(
            result.metrics["calibration_acceptance_status"],
            "FAIL",
        )
        for axis in result.metrics["axes"].values():
            self.assertTrue(axis["performance_evidence_precise"])
            self.assertFalse(axis["rmse_requirement_met"])

    def test_uncertainty_failure_is_separate_from_dataset_adequacy(self):
        def exceed_uncertainty_limit(task):
            task["d6"]["axes"]["Fz"]["calibrated_force_uncertainty"] = 0.10

        copied_examples, task_path = self.modify_task(
            exceed_uncertainty_limit
        )
        result = self.evaluate(
            dataset_path=copied_examples / "d6/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.PASS)
        self.assertEqual(result.metrics["dataset_adequacy_status"], "PASS")
        self.assertEqual(
            result.metrics["calibration_acceptance_status"],
            "FAIL",
        )
        self.assertFalse(
            result.metrics["axes"]["Fz"]["uncertainty_requirement_met"]
        )

    def test_missing_required_declaration_is_indeterminate(self):
        copied_examples, task_path = self.modify_task(
            lambda task: task["d6"].update(confidence_level=None)
        )
        result = self.evaluate(
            dataset_path=copied_examples / "d6/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.INDETERMINATE)
        self.assertIn(
            "task.d6.confidence_level",
            result.missing_evidence,
        )

    def test_d5_failure_blocks_d6(self):
        copied_examples, task_path = self.modify_task(
            lambda task: task["d5"].update(
                split_frozen_before_development=False
            )
        )
        result = self.evaluate(
            dataset_path=copied_examples / "d6/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.INDETERMINATE)
        self.assertEqual(result.context["d5_status"], "FAIL")
        self.assertIn("prerequisite.D5_PASS", result.missing_evidence)

    def test_too_few_test_runs_for_declared_bootstrap_fails(self):
        copied_examples, task_path = self.modify_task(
            lambda task: task["d6"].update(minimum_bootstrap_units=5)
        )
        result = self.evaluate(
            dataset_path=copied_examples / "d6/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "insufficient_independent_test_units_for_bootstrap",
            result.metrics["violations_by_code"],
        )

    def test_seed_makes_bootstrap_evidence_reproducible(self):
        first = self.evaluate()
        second = self.evaluate()

        self.assertEqual(first.metrics["axes"], second.metrics["axes"])

    def test_missing_d6_configuration_is_indeterminate(self):
        result = self.evaluate(
            dataset_path=D5_DEMO / "data.csv",
            task_path=D5_DEMO / "task.yaml",
        )

        self.assertEqual(result.status, CriterionStatus.INDETERMINATE)
        self.assertIn("task.d6", result.missing_evidence)

    def test_region_with_too_few_supporting_runs_fails(self):
        def move_region_outside_test_domain(task):
            task["d6"]["regions"][0]["dimensions"]["Fx"] = {
                "minimum": 10.0
            }

        copied_examples, task_path = self.modify_task(
            move_region_outside_test_domain
        )
        result = self.evaluate(
            dataset_path=copied_examples / "d6/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "insufficient_regional_test_units",
            result.metrics["violations_by_code"],
        )


if __name__ == "__main__":
    unittest.main()
