import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from calibration_adequacy.config import load_task_bundle
from calibration_adequacy.criteria import evaluate_d0, evaluate_d1, evaluate_d7
from calibration_adequacy.models import CriterionStatus


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
D1_DEMO = REPOSITORY_ROOT / "software/examples/d1/synthetic_demo"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class D7ReproducibilityProvenanceTests(unittest.TestCase):
    def evidence_package(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        package = Path(temporary_directory.name) / "package"
        shutil.copytree(D1_DEMO, package)
        task_path = package / "task.yaml"
        task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        task["d7"] = {
            "evidence_manifest_path": "evidence.yaml",
            "evidence_package_id": "synthetic_d7_package",
            "required_file_roles": [
                "raw_data",
                "task_config",
                "sensor_profile",
                "reference_profile",
                "setup_profile",
                "criterion_result",
            ],
            "required_criterion_results": ["D0", "D1"],
            "numerical_absolute_tolerance": 1.0e-12,
            "numerical_relative_tolerance": 1.0e-12,
        }
        task_path.write_text(
            yaml.safe_dump(task, sort_keys=False),
            encoding="utf-8",
        )

        bundle = load_task_bundle(task_path)
        results = {
            "D0": evaluate_d0(bundle),
            "D1": evaluate_d1(package / "data.csv", bundle),
        }
        result_paths = {}
        for criterion_id, result in results.items():
            result_path = package / f"{criterion_id.lower()}-result.json"
            result_path.write_text(
                json.dumps(result.model_dump(mode="json"), indent=2) + "\n",
                encoding="utf-8",
            )
            result_paths[criterion_id] = result_path.name

        role_by_name = {
            "data.csv": "raw_data",
            "task.yaml": "task_config",
            "sensor.yaml": "sensor_profile",
            "reference.yaml": "reference_profile",
            "setup.yaml": "setup_profile",
            "d0-result.json": "criterion_result",
            "d1-result.json": "criterion_result",
        }
        files = [
            {
                "path": name,
                "role": role,
                "sha256": sha256(package / name),
            }
            for name, role in role_by_name.items()
        ]
        manifest = {
            "schema_version": "0.1",
            "package_id": "synthetic_d7_package",
            "files": files,
            "software_commit": "abcdef0",
            "dependency_snapshot_id": "synthetic_dependencies_v1",
            "preprocessing_procedure_id": "synthetic_no_preprocessing_v1",
            "exclusion_record_id": "synthetic_no_exclusions_v1",
            "partition_manifest_id": "not_applicable_d1_only",
            "random_seeds": {},
            "criterion_results": result_paths,
        }
        (package / "evidence.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False),
            encoding="utf-8",
        )
        return package, task_path

    def test_intact_package_reproduces_recorded_results(self):
        package, task_path = self.evidence_package()

        result = evaluate_d7(
            package / "data.csv",
            load_task_bundle(task_path),
        )

        self.assertEqual(result.status, CriterionStatus.PASS)
        self.assertTrue(result.metrics["reproduction"]["D0"]["reproduced"])
        self.assertTrue(result.metrics["reproduction"]["D1"]["reproduced"])

    def test_hash_mismatch_fails(self):
        package, task_path = self.evidence_package()
        with (package / "data.csv").open("a", encoding="utf-8") as stream:
            stream.write("\n")

        result = evaluate_d7(
            package / "data.csv",
            load_task_bundle(task_path),
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "evidence_hash_mismatch",
            result.metrics["violations_by_code"],
        )

    def test_unmanifested_recorded_result_fails(self):
        package, task_path = self.evidence_package()
        manifest_path = package / "evidence.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        manifest["files"] = [
            record
            for record in manifest["files"]
            if record["path"] != "d1-result.json"
        ]
        manifest_path.write_text(
            yaml.safe_dump(manifest, sort_keys=False),
            encoding="utf-8",
        )

        result = evaluate_d7(
            package / "data.csv",
            load_task_bundle(task_path),
        )

        self.assertEqual(result.status, CriterionStatus.FAIL)
        self.assertIn(
            "active_input_not_manifested",
            result.metrics["violations_by_code"],
        )

    def test_missing_manifest_declaration_is_indeterminate(self):
        bundle = load_task_bundle(D1_DEMO / "task.yaml")
        result = evaluate_d7(D1_DEMO / "data.csv", bundle)

        self.assertEqual(result.status, CriterionStatus.INDETERMINATE)
        self.assertIn("task.d7", result.missing_evidence)


if __name__ == "__main__":
    unittest.main()
