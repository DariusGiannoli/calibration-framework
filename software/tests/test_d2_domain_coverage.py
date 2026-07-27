import csv
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from calibration_adequacy.config import load_task_bundle
from calibration_adequacy.criteria import evaluate_d2
from calibration_adequacy.models import CriterionStatus


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
D1_DEMO = REPOSITORY_ROOT / "software/examples/d1/synthetic_demo"
D2_DEMO = REPOSITORY_ROOT / "software/examples/d2/synthetic_demo"


class D2DomainCoverageTests(unittest.TestCase):
    def evaluate(self, dataset_path=None, task_path=None):
        dataset = dataset_path or D2_DEMO / "data.csv"
        task = task_path or D2_DEMO / "task.yaml"
        return evaluate_d2(dataset, load_task_bundle(task))

    def temporary_demo_copy(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        copied_examples = root / "examples"
        shutil.copytree(REPOSITORY_ROOT / "software/examples", copied_examples)
        return copied_examples / "d2/synthetic_demo"

    def write_rows(self, rows, fieldnames):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "data.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def read_demo_rows(self):
        with (D2_DEMO / "data.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            return list(reader), list(reader.fieldnames or [])

    def test_complete_grid_passes_with_zero_fill_distance(self):
        result = self.evaluate()

        self.assertEqual(result.status, CriterionStatus.PASS)
        self.assertEqual(result.context["d1_status"], "PASS")
        self.assertEqual(result.metrics["samples_used"], 27)
        self.assertEqual(result.metrics["grid_point_count"], 27)
        self.assertAlmostEqual(result.metrics["estimated_fill_distance"], 0.0)

    def test_sparse_center_only_dataset_fails_with_expected_distance(self):
        rows, fieldnames = self.read_demo_rows()
        center = [
            row
            for row in rows
            if row["Fx_ref"] == row["Fy_ref"] == row["Fz_ref"] == "0.0"
        ]
        dataset = self.write_rows(center, fieldnames)

        result = self.evaluate(dataset_path=dataset)

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertAlmostEqual(
            result.metrics["estimated_fill_distance"],
            math.sqrt(3) / 2,
        )
        self.assertIn(
            "fill_distance_exceeded",
            result.metrics["violations_by_code"],
        )
        self.assertEqual(result.violations[0].code, "fill_distance_exceeded")

    def test_missing_d2_configuration_is_indeterminate(self):
        result = self.evaluate(
            dataset_path=D1_DEMO / "data.csv",
            task_path=D1_DEMO / "task.yaml",
        )

        self.assertEqual(result.status, CriterionStatus.INDETERMINATE)
        self.assertIn("task.d2", result.missing_evidence)

    def test_missing_domain_declaration_is_indeterminate(self):
        copied_demo = self.temporary_demo_copy()
        task_path = copied_demo / "task.yaml"
        task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        task["d2"]["domain"]["Fz"]["grid_points"] = None
        task_path.write_text(
            yaml.safe_dump(task, sort_keys=False),
            encoding="utf-8",
        )

        result = self.evaluate(
            dataset_path=copied_demo / "data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.INDETERMINATE)
        self.assertIn("task.d2.domain.Fz.grid_points", result.missing_evidence)

    def test_d1_failure_blocks_d2(self):
        rows, fieldnames = self.read_demo_rows()
        rows[1]["reference_time_s"] = "0.100"
        dataset = self.write_rows(rows, fieldnames)

        result = self.evaluate(dataset_path=dataset)

        self.assertEqual(result.status, CriterionStatus.INDETERMINATE)
        self.assertEqual(result.context["d1_status"], "FAIL")
        self.assertIn("prerequisite.D1_PASS", result.missing_evidence)

    def test_reference_points_are_rotated_into_the_sensor_frame(self):
        copied_demo = self.temporary_demo_copy()
        setup_path = copied_demo.parents[1] / "d1/synthetic_demo/setup.yaml"
        setup = yaml.safe_load(setup_path.read_text(encoding="utf-8"))
        setup["reference_to_sensor_rotation"] = [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        setup_path.write_text(
            yaml.safe_dump(setup, sort_keys=False),
            encoding="utf-8",
        )

        rows, fieldnames = self.read_demo_rows()
        selected = [
            row
            for row in rows
            if row["Fx_ref"] == "1.0"
            and row["Fy_ref"] == "0.0"
            and row["Fz_ref"] == "0.0"
        ]
        dataset = self.write_rows(selected, fieldnames)

        result = self.evaluate(
            dataset_path=dataset,
            task_path=copied_demo / "task.yaml",
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertAlmostEqual(result.metrics["achieved_minimum"]["Fx"], 0.0)
        self.assertAlmostEqual(result.metrics["achieved_minimum"]["Fy"], 1.0)
        self.assertAlmostEqual(result.metrics["achieved_minimum"]["Fz"], 0.0)


if __name__ == "__main__":
    unittest.main()
