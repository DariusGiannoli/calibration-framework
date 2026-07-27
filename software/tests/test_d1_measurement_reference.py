import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from calibration_adequacy.config import load_task_bundle
from calibration_adequacy.criteria import evaluate_d1
from calibration_adequacy.models import CriterionStatus


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DEMO = REPOSITORY_ROOT / "software/examples/d1/synthetic_demo"
TRIAXIAL_TASK = REPOSITORY_ROOT / "software/configs/tasks/triaxial_t1.yaml"


class D1MeasurementReferenceTests(unittest.TestCase):
    def evaluate_synthetic(self, dataset_path=None, task_path=None):
        task = task_path or SYNTHETIC_DEMO / "task.yaml"
        dataset = dataset_path or SYNTHETIC_DEMO / "data.csv"
        return evaluate_d1(dataset, load_task_bundle(task))

    def write_modified_csv(self, rows, fieldnames):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "data.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def read_demo_rows(self):
        with (SYNTHETIC_DEMO / "data.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            return list(reader), list(reader.fieldnames or [])

    def test_complete_synthetic_dataset_passes(self):
        result = self.evaluate_synthetic()

        self.assertEqual(result.status, CriterionStatus.PASS)
        self.assertEqual(result.context["task_id"], "synthetic_triaxial_t1")
        self.assertEqual(result.metrics["rows_evaluated"], 3)
        self.assertEqual(result.metrics["invalid_rows"], 0)
        self.assertEqual(result.missing_evidence, [])
        self.assertEqual(result.violations, [])

    def test_real_templates_are_indeterminate_until_values_are_supplied(self):
        result = self.evaluate_synthetic(task_path=TRIAXIAL_TASK)

        self.assertEqual(result.status, CriterionStatus.INDETERMINATE)
        self.assertIn("setup.maximum_time_offset_s", result.missing_evidence)
        self.assertIn(
            "reference_profile.Fx.expanded_uncertainty",
            result.missing_evidence,
        )
        self.assertEqual(result.metrics["total_violations"], 0)

    def test_excessive_time_offset_fails(self):
        rows, fieldnames = self.read_demo_rows()
        rows[1]["reference_time_s"] = "0.050"
        dataset = self.write_modified_csv(rows, fieldnames)

        result = self.evaluate_synthetic(dataset_path=dataset)

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "synchronization_exceeded",
            result.metrics["violations_by_code"],
        )

    def test_out_of_range_measurement_fails(self):
        rows, fieldnames = self.read_demo_rows()
        rows[1]["c2"] = "12.0"
        dataset = self.write_modified_csv(rows, fieldnames)

        result = self.evaluate_synthetic(dataset_path=dataset)

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "measurement_out_of_range",
            result.metrics["violations_by_code"],
        )

    def test_missing_required_channel_fails(self):
        rows, fieldnames = self.read_demo_rows()
        fieldnames.remove("Fz_ref")
        for row in rows:
            row.pop("Fz_ref")
        dataset = self.write_modified_csv(rows, fieldnames)

        result = self.evaluate_synthetic(dataset_path=dataset)

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "missing_required_column",
            result.metrics["violations_by_code"],
        )

    def test_reference_uncertainty_above_task_limit_fails(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        copied_demo = Path(temporary_directory.name) / "demo"
        shutil.copytree(SYNTHETIC_DEMO, copied_demo)

        reference_path = copied_demo / "reference.yaml"
        reference = yaml.safe_load(reference_path.read_text(encoding="utf-8"))
        reference["channels"]["Fx"]["expanded_uncertainty"] = 0.1
        reference_path.write_text(
            yaml.safe_dump(reference, sort_keys=False),
            encoding="utf-8",
        )

        result = self.evaluate_synthetic(
            dataset_path=copied_demo / "data.csv",
            task_path=copied_demo / "task.yaml",
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "reference_uncertainty_exceeded",
            result.metrics["violations_by_code"],
        )

    def test_invalid_coordinate_rotation_fails(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        copied_demo = Path(temporary_directory.name) / "demo"
        shutil.copytree(SYNTHETIC_DEMO, copied_demo)

        setup_path = copied_demo / "setup.yaml"
        setup = yaml.safe_load(setup_path.read_text(encoding="utf-8"))
        setup["reference_to_sensor_rotation"] = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        setup_path.write_text(
            yaml.safe_dump(setup, sort_keys=False),
            encoding="utf-8",
        )

        result = self.evaluate_synthetic(
            dataset_path=copied_demo / "data.csv",
            task_path=copied_demo / "task.yaml",
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "invalid_coordinate_rotation",
            result.metrics["violations_by_code"],
        )

    def test_non_monotonic_timestamps_fail(self):
        rows, fieldnames = self.read_demo_rows()
        rows[2]["sensor_time_s"] = "0.005"
        dataset = self.write_modified_csv(rows, fieldnames)

        result = self.evaluate_synthetic(dataset_path=dataset)

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "non_monotonic_timestamp",
            result.metrics["violations_by_code"],
        )

    def test_missing_exclusion_policy_is_indeterminate(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        copied_demo = Path(temporary_directory.name) / "demo"
        shutil.copytree(SYNTHETIC_DEMO, copied_demo)
        task_path = copied_demo / "task.yaml"
        task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        task["d1"]["invalid_observation_policy_id"] = None
        task_path.write_text(
            yaml.safe_dump(task, sort_keys=False),
            encoding="utf-8",
        )

        result = self.evaluate_synthetic(
            dataset_path=copied_demo / "data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.INDETERMINATE)
        self.assertIn(
            "task.d1.invalid_observation_policy_id",
            result.missing_evidence,
        )

    def test_unreviewed_exclusions_fail(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        copied_demo = Path(temporary_directory.name) / "demo"
        shutil.copytree(SYNTHETIC_DEMO, copied_demo)
        task_path = copied_demo / "task.yaml"
        task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        task["d1"]["exclusions_reviewed"] = False
        task_path.write_text(
            yaml.safe_dump(task, sort_keys=False),
            encoding="utf-8",
        )

        result = self.evaluate_synthetic(
            dataset_path=copied_demo / "data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "exclusions_not_reviewed",
            result.metrics["violations_by_code"],
        )


if __name__ == "__main__":
    unittest.main()
