import tempfile
import unittest
from pathlib import Path

import yaml

from calibration_adequacy.config import (
    load_declaration_register,
    load_task_bundle,
)
from calibration_adequacy.declaration_register import (
    audit_declaration_register,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TASK_PATH = (
    REPOSITORY_ROOT / "software/configs/tasks/triaxial_t1.yaml"
)
REGISTER_PATH = (
    REPOSITORY_ROOT
    / "software/configs/tasks/triaxial_t1_declarations.yaml"
)


class DeclarationRegisterTests(unittest.TestCase):
    def test_triaxial_register_matches_live_d0_snapshot(self):
        audit = audit_declaration_register(
            load_task_bundle(TASK_PATH),
            load_declaration_register(REGISTER_PATH),
        )

        self.assertTrue(audit.aligned)
        self.assertEqual(audit.actual_missing_count, 37)
        self.assertEqual(audit.registered_count, 37)
        self.assertTrue(audit.order_matches)
        self.assertEqual(
            audit.classification_counts,
            {
                "1_physical_or_application_requirement": 18,
                "2_hardware_or_reference_specification": 0,
                "3_experimental_design_choice": 9,
                (
                    "4_statistical_threshold_requiring_simulation_or_"
                    "pilot_justification"
                ): 7,
                "5_value_evaluated_after_acquisition": 3,
            },
        )
        self.assertEqual(
            audit.resolution_stage_counts,
            {
                "after_acquisition": 3,
                "before_final_acquisition": 15,
                "before_trajectory_simulation": 19,
            },
        )

    def test_audit_detects_an_unclassified_live_declaration(self):
        raw = yaml.safe_load(REGISTER_PATH.read_text(encoding="utf-8"))
        removed_path = raw["entries"].pop()["path"]
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        register_path = Path(temporary_directory.name) / "register.yaml"
        register_path.write_text(
            yaml.safe_dump(raw, sort_keys=False),
            encoding="utf-8",
        )

        audit = audit_declaration_register(
            load_task_bundle(TASK_PATH),
            load_declaration_register(register_path),
        )

        self.assertFalse(audit.aligned)
        self.assertEqual(audit.missing_from_register, [removed_path])
        self.assertEqual(audit.no_longer_missing, [])


if __name__ == "__main__":
    unittest.main()
