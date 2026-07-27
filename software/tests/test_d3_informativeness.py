import csv
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from calibration_adequacy.config import load_task_bundle
from calibration_adequacy.criteria import evaluate_d3
from calibration_adequacy.models import CriterionStatus


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
D1_DEMO = REPOSITORY_ROOT / "software/examples/d1/synthetic_demo"
D3_DEMO = REPOSITORY_ROOT / "software/examples/d3/synthetic_demo"


class D3InformativenessTests(unittest.TestCase):
    def evaluate(self, dataset_path=None, task_path=None):
        dataset = dataset_path or D3_DEMO / "data.csv"
        task = task_path or D3_DEMO / "task.yaml"
        return evaluate_d3(dataset, load_task_bundle(task))

    def temporary_examples_copy(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        copied_examples = root / "examples"
        shutil.copytree(REPOSITORY_ROOT / "software/examples", copied_examples)
        return copied_examples

    def read_demo_rows(self):
        with (D3_DEMO / "data.csv").open(
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

    def test_independent_full_factorial_inputs_pass(self):
        result = self.evaluate()

        self.assertEqual(result.status, CriterionStatus.PASS)
        self.assertEqual(result.context["d1_status"], "PASS")
        self.assertEqual(result.metrics["samples_used"], 27)
        self.assertEqual(result.metrics["design_matrix_rank"], 4)
        self.assertEqual(result.metrics["sensitivity_matrix_rank"], 12)
        self.assertEqual(result.metrics["total_parameter_count"], 12)
        self.assertAlmostEqual(
            result.metrics["condition_number"],
            math.sqrt(1.5),
        )

    def test_perfectly_collinear_inputs_fail_rank(self):
        rows, fieldnames = self.read_demo_rows()
        for row in rows:
            row["c2"] = row["c1"]
            row["c3"] = row["c1"]
        dataset = self.write_rows(rows, fieldnames)

        result = self.evaluate(dataset_path=dataset)

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertLess(result.metrics["design_matrix_rank"], 4)
        self.assertTrue(result.metrics["condition_number_is_infinite"])
        self.assertIn("rank_deficient", result.metrics["violations_by_code"])

    def test_nearly_collinear_inputs_fail_conditioning_while_full_rank(self):
        rows, fieldnames = self.read_demo_rows()
        for row in rows:
            c1 = float(row["c1"])
            original_c3 = float(row["c3"])
            row["c3"] = str(c1 + 1.0e-6 * original_c3)
        dataset = self.write_rows(rows, fieldnames)

        result = self.evaluate(dataset_path=dataset)

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertEqual(result.metrics["design_matrix_rank"], 4)
        self.assertGreater(result.metrics["condition_number"], 2.0)
        self.assertNotIn("rank_deficient", result.metrics["violations_by_code"])
        self.assertIn(
            "condition_number_exceeded",
            result.metrics["violations_by_code"],
        )

    def test_missing_d3_configuration_is_indeterminate(self):
        result = self.evaluate(
            dataset_path=D1_DEMO / "data.csv",
            task_path=D1_DEMO / "task.yaml",
        )

        self.assertEqual(result.status, CriterionStatus.INDETERMINATE)
        self.assertIn("task.d3", result.missing_evidence)

    def test_missing_required_declaration_is_indeterminate(self):
        copied_examples = self.temporary_examples_copy()
        task_path = copied_examples / "d3/synthetic_demo/task.yaml"
        task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        task["d3"]["normalization"]["c3"]["maximum"] = None
        task["d3"]["relative_rank_tolerance"] = None
        task_path.write_text(
            yaml.safe_dump(task, sort_keys=False),
            encoding="utf-8",
        )

        result = self.evaluate(
            dataset_path=copied_examples / "d3/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.INDETERMINATE)
        self.assertIn(
            "task.d3.normalization.c3.maximum",
            result.missing_evidence,
        )
        self.assertIn(
            "task.d3.relative_rank_tolerance",
            result.missing_evidence,
        )

    def test_d1_failure_blocks_d3(self):
        rows, fieldnames = self.read_demo_rows()
        rows[1]["reference_time_s"] = "0.100"
        dataset = self.write_rows(rows, fieldnames)

        result = self.evaluate(dataset_path=dataset)

        self.assertEqual(result.status, CriterionStatus.INDETERMINATE)
        self.assertEqual(result.context["d1_status"], "FAIL")
        self.assertIn("prerequisite.D1_PASS", result.missing_evidence)

    def test_output_dimension_must_match_reference_measurand(self):
        copied_examples = self.temporary_examples_copy()
        task_path = copied_examples / "d3/synthetic_demo/task.yaml"
        task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        task["d3"]["output_dimension"] = 2
        task_path.write_text(
            yaml.safe_dump(task, sort_keys=False),
            encoding="utf-8",
        )

        result = self.evaluate(
            dataset_path=copied_examples / "d3/synthetic_demo/data.csv",
            task_path=task_path,
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "output_dimension_mismatch",
            result.metrics["violations_by_code"],
        )


if __name__ == "__main__":
    unittest.main()
