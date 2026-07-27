import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from calibration_adequacy.config import load_task_bundle
from calibration_adequacy.criteria import evaluate_all, evaluate_d0
from calibration_adequacy.models import CriterionStatus


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
D6_DEMO = REPOSITORY_ROOT / "software/examples/d6/synthetic_demo"


class D0AndOverallAssessmentTests(unittest.TestCase):
    def complete_task_copy(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        examples = root / "examples"
        shutil.copytree(REPOSITORY_ROOT / "software/examples", examples)
        task_path = examples / "d6/synthetic_demo/task.yaml"
        task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        task["claim"] = {
            "sensor_inputs": ["c1", "c2", "c3"],
            "reference_outputs": ["Fx", "Fy", "Fz"],
            "operating_domain": ["Fx", "Fy", "Fz"],
            "operating_conditions": [],
            "model_family": "affine",
            "performance_metrics": [
                "rmse",
                "calibrated_force_uncertainty",
            ],
            "generalization": {
                "target_description": "new runs of the same sensor",
                "independent_unit": "run",
                "sensor_scope": "same_sensor",
                "operating_conditions": [],
            },
        }
        task["d2"] = {
            "axes": ["Fx", "Fy", "Fz"],
            "domain": {
                axis: {
                    "minimum": -1.1,
                    "maximum": 1.1,
                    "grid_points": 2,
                }
                for axis in ("Fx", "Fy", "Fz")
            },
            "conditions": {},
            "excluded_regions": [],
            "maximum_fill_distance": 2.0,
        }
        run_ids = [
            "development_01",
            "development_02",
            "development_03",
            "test_01",
            "test_02",
            "test_03",
            "test_04",
        ]
        task["d4"] = {
            "independent_unit": "run",
            "dependence_method_id": "fixed_lag_autocorrelation_ess_v1",
            "stationarity_reviewed": True,
            "signals": [
                {
                    "source": source,
                    "channel": channel,
                    "maximum_autocorrelation_lag": 1,
                }
                for source, channels in (
                    ("sensor", ("c1", "c2", "c3")),
                    ("reference", ("Fx", "Fy", "Fz")),
                )
                for channel in channels
            ],
            "minimum_independent_runs": 7,
            "minimum_runs_per_configuration": {},
            "minimum_effective_sample_size": 1.0,
            "initialization_procedure_id": "synthetic_initialization_v1",
            "zeroing_procedure_id": "synthetic_zeroing_v1",
            "run_evidence": [
                {
                    "run_id": run_id,
                    "acquisition_id": f"acquisition_{index:02d}",
                    "separate_acquisition": True,
                    "initialization_completed": True,
                    "zeroing_completed": True,
                }
                for index, run_id in enumerate(run_ids, start=1)
            ],
        }
        task["assessment"] = {
            "criteria": {
                criterion: {"applicable": True}
                for criterion in ("D1", "D2", "D3", "D4", "D5", "D6")
            }
        }
        task["assessment"]["criteria"]["D7"] = {
            "applicable": False,
            "reason": "synthetic aggregation test has no evidence package",
        }
        task_path.write_text(
            yaml.safe_dump(task, sort_keys=False),
            encoding="utf-8",
        )
        return examples, task_path

    def test_complete_claim_passes_d0(self):
        _, task_path = self.complete_task_copy()
        result = evaluate_d0(load_task_bundle(task_path))

        self.assertEqual(result.status, CriterionStatus.PASS)
        self.assertEqual(result.missing_evidence, [])

    def test_claim_mapping_contradiction_fails_d0(self):
        _, task_path = self.complete_task_copy()
        task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        task["claim"]["sensor_inputs"] = ["c1", "c2"]
        task_path.write_text(
            yaml.safe_dump(task, sort_keys=False),
            encoding="utf-8",
        )

        result = evaluate_d0(load_task_bundle(task_path))

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "sensor_input_mismatch",
            result.metrics["violations_by_code"],
        )

    def test_overall_aggregation_excludes_justified_not_applicable(self):
        examples, task_path = self.complete_task_copy()
        result = evaluate_all(
            examples / "d6/synthetic_demo/data.csv",
            load_task_bundle(task_path),
        )

        self.assertEqual(result.status, CriterionStatus.PASS)
        self.assertEqual(
            result.criteria["D7"].status,
            CriterionStatus.NOT_APPLICABLE,
        )
        self.assertEqual(
            result.calibration_acceptance_status,
            CriterionStatus.PASS,
        )


if __name__ == "__main__":
    unittest.main()
