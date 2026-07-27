from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Type

from pydantic import BaseModel

from .config import ConfigurationError, load_task_bundle
from .criteria import (
    evaluate_d1,
    evaluate_d2,
    evaluate_d3,
    evaluate_d4,
    evaluate_d5,
)
from .models import (
    CriterionStatus,
    InstrumentProfile,
    SetupProfile,
    TaskConfig,
)


def _write_schemas(output_directory: Path) -> int:
    output_directory.mkdir(parents=True, exist_ok=True)
    models: Dict[str, Type[BaseModel]] = {
        "instrument-profile.schema.json": InstrumentProfile,
        "setup-profile.schema.json": SetupProfile,
        "task.schema.json": TaskConfig,
    }
    for filename, model in models.items():
        target = output_directory / filename
        target.write_text(
            json.dumps(model.model_json_schema(), indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"schemas written to {output_directory}")
    return 0


def _evaluate_d1(arguments: argparse.Namespace) -> int:
    bundle = load_task_bundle(arguments.task)
    result = evaluate_d1(arguments.dataset, bundle)
    rendered = json.dumps(result.model_dump(mode="json"), indent=2) + "\n"

    if arguments.output:
        output_path = Path(arguments.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(f"D1 evidence report: {output_path}")
    else:
        print(rendered, end="")

    return {
        CriterionStatus.PASS: 0,
        CriterionStatus.FAIL: 1,
        CriterionStatus.INDETERMINATE: 2,
    }[result.status]


def _evaluate_d2(arguments: argparse.Namespace) -> int:
    bundle = load_task_bundle(arguments.task)
    result = evaluate_d2(arguments.dataset, bundle)
    rendered = json.dumps(result.model_dump(mode="json"), indent=2) + "\n"

    if arguments.output:
        output_path = Path(arguments.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(f"D2 evidence report: {output_path}")
    else:
        print(rendered, end="")

    return {
        CriterionStatus.PASS: 0,
        CriterionStatus.FAIL: 1,
        CriterionStatus.INDETERMINATE: 2,
    }[result.status]


def _evaluate_d3(arguments: argparse.Namespace) -> int:
    bundle = load_task_bundle(arguments.task)
    result = evaluate_d3(arguments.dataset, bundle)
    rendered = json.dumps(result.model_dump(mode="json"), indent=2) + "\n"

    if arguments.output:
        output_path = Path(arguments.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(f"D3 evidence report: {output_path}")
    else:
        print(rendered, end="")

    return {
        CriterionStatus.PASS: 0,
        CriterionStatus.FAIL: 1,
        CriterionStatus.INDETERMINATE: 2,
    }[result.status]


def _evaluate_d4(arguments: argparse.Namespace) -> int:
    bundle = load_task_bundle(arguments.task)
    result = evaluate_d4(arguments.dataset, bundle)
    rendered = json.dumps(result.model_dump(mode="json"), indent=2) + "\n"

    if arguments.output:
        output_path = Path(arguments.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(f"D4 evidence report: {output_path}")
    else:
        print(rendered, end="")

    return {
        CriterionStatus.PASS: 0,
        CriterionStatus.FAIL: 1,
        CriterionStatus.INDETERMINATE: 2,
    }[result.status]


def _evaluate_d5(arguments: argparse.Namespace) -> int:
    bundle = load_task_bundle(arguments.task)
    result = evaluate_d5(arguments.dataset, bundle)
    rendered = json.dumps(result.model_dump(mode="json"), indent=2) + "\n"

    if arguments.output:
        output_path = Path(arguments.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(f"D5 evidence report: {output_path}")
    else:
        print(rendered, end="")

    return {
        CriterionStatus.PASS: 0,
        CriterionStatus.FAIL: 1,
        CriterionStatus.INDETERMINATE: 2,
    }[result.status]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibration-adequacy",
        description="Evaluate calibration dataset adequacy criteria.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    d1_parser = subparsers.add_parser(
        "evaluate-d1",
        help="Evaluate D1 measurement and reference validity for a CSV dataset.",
    )
    d1_parser.add_argument("--task", required=True, help="Path to task YAML.")
    d1_parser.add_argument("--dataset", required=True, help="Path to dataset CSV.")
    d1_parser.add_argument("--output", help="Optional JSON evidence-report path.")
    d1_parser.set_defaults(handler=_evaluate_d1)

    d2_parser = subparsers.add_parser(
        "evaluate-d2",
        help="Evaluate D2 domain coverage for a CSV dataset.",
    )
    d2_parser.add_argument("--task", required=True, help="Path to task YAML.")
    d2_parser.add_argument("--dataset", required=True, help="Path to dataset CSV.")
    d2_parser.add_argument("--output", help="Optional JSON evidence-report path.")
    d2_parser.set_defaults(handler=_evaluate_d2)

    d3_parser = subparsers.add_parser(
        "evaluate-d3",
        help="Evaluate D3 informativeness and identifiability for a CSV dataset.",
    )
    d3_parser.add_argument("--task", required=True, help="Path to task YAML.")
    d3_parser.add_argument("--dataset", required=True, help="Path to dataset CSV.")
    d3_parser.add_argument("--output", help="Optional JSON evidence-report path.")
    d3_parser.set_defaults(handler=_evaluate_d3)

    d4_parser = subparsers.add_parser(
        "evaluate-d4",
        help="Evaluate D4 independent replication and dependence for a CSV dataset.",
    )
    d4_parser.add_argument("--task", required=True, help="Path to task YAML.")
    d4_parser.add_argument("--dataset", required=True, help="Path to dataset CSV.")
    d4_parser.add_argument("--output", help="Optional JSON evidence-report path.")
    d4_parser.set_defaults(handler=_evaluate_d4)

    d5_parser = subparsers.add_parser(
        "evaluate-d5",
        help="Evaluate D5 leakage-resistant validation for a CSV dataset.",
    )
    d5_parser.add_argument("--task", required=True, help="Path to task YAML.")
    d5_parser.add_argument("--dataset", required=True, help="Path to dataset CSV.")
    d5_parser.add_argument("--output", help="Optional JSON evidence-report path.")
    d5_parser.set_defaults(handler=_evaluate_d5)

    schema_parser = subparsers.add_parser(
        "write-schemas",
        help="Write JSON Schemas for future GUI form generation.",
    )
    schema_parser.add_argument("--output-dir", required=True, type=Path)
    schema_parser.set_defaults(
        handler=lambda arguments: _write_schemas(arguments.output_dir)
    )
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        exit_code = arguments.handler(arguments)
    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        exit_code = 3
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
