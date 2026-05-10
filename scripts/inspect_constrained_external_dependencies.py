"""Inspect optional external constrained-comparator dependencies.

This script is intentionally inspection-only:
- no dependency installation
- no optimizer execution
- no benchmark execution
- no default NSGA-II or constrained NSGA-II code changes
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import platform
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback, not expected here
    tomllib = None  # type: ignore[assignment]


ALLOWED_INSPECTION_DECISIONS = {
    "pymoo_ready_for_wrapper_planning",
    "pymoo_dependency_missing",
    "pymoo_api_review_needed",
    "deap_secondary_hold",
    "implementation_not_ready",
    "fix_required",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _fresh_path(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _safe_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _signature_or_error(obj: object) -> dict[str, Any]:
    try:
        signature = inspect.signature(obj)
    except Exception as exc:  # pragma: no cover - defensive for optional APIs
        return {"available": False, "error": str(exc), "signature": None, "parameters": []}
    return {
        "available": True,
        "error": None,
        "signature": str(signature),
        "parameters": list(signature.parameters),
    }


def _import_symbol(path: str) -> tuple[object | None, dict[str, Any]]:
    module_name, _, symbol_name = path.rpartition(".")
    try:
        module = importlib.import_module(module_name)
        symbol = getattr(module, symbol_name)
    except Exception as exc:
        return None, {
            "path": path,
            "available": False,
            "error": str(exc),
        }
    return symbol, {
        "path": path,
        "available": True,
        "error": None,
    }


def _inspect_pymoo() -> dict[str, Any]:
    spec = importlib.util.find_spec("pymoo")
    installed = spec is not None
    version = _safe_version("pymoo")
    result: dict[str, Any] = {
        "installed": installed,
        "version": version,
        "importlib_metadata_version": version,
        "import_status": "dependency_missing" if not installed else "api_import_pending",
        "api_symbols": {},
        "signatures": {},
        "constraint_api_notes": {
            "constructor_parameter_candidates": [],
            "output_key_candidates": ["F", "G", "H"],
            "constraint_sign_convention": "must_be_verified_during_implementation",
            "notes": [
                "F is the objective output key candidate.",
                "G/H are inequality/equality constraint output key candidates in common pymoo versions, but local API semantics must be verified before implementation.",
            ],
        },
        "evaluation_budget_notes": {
            "termination_candidate": "termination=('n_eval', budget)",
            "actual_evaluation_count_candidates": [
                "problem.evaluations_counter_in_wrapper",
                "algorithm.evaluator.n_eval",
                "result.algorithm.evaluator.n_eval",
            ],
            "policy": "exact actual_evaluations must be verified during implementation",
        },
        "seed_notes": {
            "seed_parameter_candidate": "pymoo.optimize.minimize(..., seed=seed)",
            "policy": "seed handling must be recorded in future artifacts",
        },
        "probe": {
            "dummy_problem_subclass": "not_attempted",
            "optimizer_execution": "forbidden",
            "benchmark_execution": "forbidden",
        },
        "risks": [
            "constraint sign convention must be verified during implementation",
            "actual evaluation accounting must be verified during implementation",
            "operator family difference warning required",
            "API/version mismatch possible because pymoo is optional",
        ],
        "recommended_status": "dependency_missing" if not installed else "api_import_failed",
        "warnings": [],
        "failures": [],
    }
    if not installed:
        result["warnings"].append("pymoo is not installed; future comparator should record skip artifact")
        return result

    try:
        module = importlib.import_module("pymoo")
        result["module_version"] = str(getattr(module, "__version__", "unknown"))
    except Exception as exc:
        result["import_status"] = "api_import_failed"
        result["failures"].append(f"pymoo package import failed: {exc}")
        result["recommended_status"] = "api_import_failed"
        return result

    symbols = {
        "Problem": "pymoo.core.problem.Problem",
        "NSGA2": "pymoo.algorithms.moo.nsga2.NSGA2",
        "minimize": "pymoo.optimize.minimize",
    }
    imported: dict[str, object] = {}
    for name, path in symbols.items():
        symbol, status = _import_symbol(path)
        result["api_symbols"][name] = status
        if symbol is not None:
            imported[name] = symbol
            result["signatures"][name] = _signature_or_error(symbol)

    failed_symbols = [
        name for name, status in result["api_symbols"].items() if not status["available"]
    ]
    if failed_symbols:
        result["import_status"] = "api_import_failed"
        result["recommended_status"] = "api_import_failed"
        result["failures"].append(f"pymoo API imports failed: {', '.join(failed_symbols)}")
        return result

    result["import_status"] = "imported"
    problem_params = set(result["signatures"].get("Problem", {}).get("parameters", []))
    candidates = [
        name for name in ("n_ieq_constr", "n_eq_constr", "n_constr") if name in problem_params
    ]
    result["constraint_api_notes"]["constructor_parameter_candidates"] = candidates
    result["constraint_api_notes"]["constructor_signature_interpretation"] = (
        "modern constraint parameters detected"
        if {"n_ieq_constr", "n_eq_constr"} & problem_params
        else "legacy_or_indirect_constraint_parameters_detected"
        if "n_constr" in problem_params
        else "constraint_parameters_not_visible_in_signature"
    )

    problem_cls = imported.get("Problem")
    if problem_cls is not None:
        try:
            class DummyInspectionProblem(problem_cls):  # type: ignore[misc, valid-type]
                def _evaluate(self, x, out, *args, **kwargs) -> None:  # pragma: no cover
                    out["F"] = []
                    out["G"] = []
                    out["H"] = []

            result["probe"]["dummy_problem_subclass"] = "defined_without_optimizer_execution"
            result["probe"]["dummy_problem_class_name"] = DummyInspectionProblem.__name__
        except Exception as exc:  # pragma: no cover - defensive for optional APIs
            result["probe"]["dummy_problem_subclass"] = "probe_skipped"
            result["probe"]["dummy_problem_error"] = str(exc)
            result["warnings"].append("dummy Problem subclass definition failed; hold for manual review")

    result["recommended_status"] = (
        "usable_with_version_guards"
        if not result["warnings"]
        else "hold_for_manual_review"
    )
    return result


def _inspect_deap() -> dict[str, Any]:
    spec = importlib.util.find_spec("deap")
    installed = spec is not None
    version = _safe_version("deap")
    result: dict[str, Any] = {
        "installed": installed,
        "version": version,
        "importlib_metadata_version": version,
        "import_status": "dependency_missing" if not installed else "api_import_pending",
        "api_symbols": {},
        "signatures": {},
        "constraint_api_notes": {
            "constraint_domination_builtin": "not_asserted",
            "inspection_note": "Only NSGA-II tool availability is inspected; constrained domination support is not assumed.",
        },
        "risks": [
            "DEAP constrained NSGA-II may require custom feasibility/violation comparator",
            "if custom comparator is required, DEAP may be less useful as external baseline",
            "operator family difference warning required",
        ],
        "recommended_status": "dependency_missing" if not installed else "api_import_failed",
        "warnings": [],
        "failures": [],
    }
    if not installed:
        result["warnings"].append("deap is not installed; optional secondary comparator should be skipped")
        return result

    modules = {
        "base": "deap.base",
        "creator": "deap.creator",
        "tools": "deap.tools",
    }
    imported_modules: dict[str, object] = {}
    for name, module_path in modules.items():
        try:
            imported_modules[name] = importlib.import_module(module_path)
            result["api_symbols"][name] = {"path": module_path, "available": True, "error": None}
        except Exception as exc:
            result["api_symbols"][name] = {
                "path": module_path,
                "available": False,
                "error": str(exc),
            }

    failed_modules = [
        name for name, status in result["api_symbols"].items() if not status["available"]
    ]
    if failed_modules:
        result["import_status"] = "api_import_failed"
        result["recommended_status"] = "api_import_failed"
        result["failures"].append(f"DEAP imports failed: {', '.join(failed_modules)}")
        return result

    tools = imported_modules["tools"]
    result["import_status"] = "imported"
    for symbol_name in ("selNSGA2", "selTournamentDCD"):
        symbol = getattr(tools, symbol_name, None)
        result["api_symbols"][symbol_name] = {
            "path": f"deap.tools.{symbol_name}",
            "available": symbol is not None,
            "error": None if symbol is not None else "symbol missing",
        }
        if symbol is not None:
            result["signatures"][symbol_name] = _signature_or_error(symbol)
    try:
        emo = importlib.import_module("deap.tools.emo")
        result["api_symbols"]["tools.emo"] = {
            "path": "deap.tools.emo",
            "available": True,
            "error": None,
            "public_symbols_sample": sorted(name for name in dir(emo) if not name.startswith("_"))[:30],
        }
    except Exception as exc:
        result["api_symbols"]["tools.emo"] = {
            "path": "deap.tools.emo",
            "available": False,
            "error": str(exc),
        }

    missing_tools = [
        name
        for name in ("selNSGA2", "selTournamentDCD")
        if not result["api_symbols"].get(name, {}).get("available")
    ]
    if missing_tools:
        result["recommended_status"] = "api_import_failed"
        result["failures"].append(f"DEAP NSGA-II tools missing: {', '.join(missing_tools)}")
        return result

    result["recommended_status"] = "not_recommended_as_primary"
    result["warnings"].append(
        "DEAP is secondary only unless constrained domination can be implemented without penalty/repair scope creep"
    )
    return result


def _inspect_internal_reference(project_root: Path) -> dict[str, Any]:
    comparator_path = project_root / "src/ga_lab/experiment/external_mo_comparators.py"
    pyproject_path = project_root / "pyproject.toml"
    pyproject_extras: dict[str, Any] = {}
    if tomllib is not None and pyproject_path.exists():
        try:
            data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
            pyproject_extras = dict(data.get("project", {}).get("optional-dependencies", {}))
        except Exception as exc:  # pragma: no cover - defensive
            pyproject_extras = {"parse_error": str(exc)}
    return {
        "external_mo_comparators_present": comparator_path.exists(),
        "external_mo_comparators_path": str(comparator_path),
        "optional_dependency_pattern": (
            "optional_library_status uses importlib.util.find_spec and returns skipped result "
            "when optional dependency is missing"
        ),
        "existing_skip_pattern": {
            "missing_dependency_status": "skipped",
            "success": False,
            "metadata_key": "library_status",
        },
        "pyproject_extras": pyproject_extras,
        "mo_compare_extra_present": "mo-compare" in pyproject_extras,
        "mo_compare_dependencies": pyproject_extras.get("mo-compare", []),
    }


def _inspection_decision(pymoo: dict[str, Any], deap: dict[str, Any]) -> str:
    if pymoo["recommended_status"] == "usable_with_version_guards":
        return "pymoo_ready_for_wrapper_planning"
    if pymoo["recommended_status"] == "dependency_missing":
        return "pymoo_dependency_missing"
    if pymoo["recommended_status"] in {"api_import_failed", "hold_for_manual_review"}:
        return "pymoo_api_review_needed"
    if deap["recommended_status"] in {"not_recommended_as_primary", "dependency_missing"}:
        return "deap_secondary_hold"
    return "implementation_not_ready"


def _render_markdown(payload: dict[str, Any]) -> str:
    pymoo = payload["dependencies"]["pymoo"]
    deap = payload["dependencies"]["deap"]
    internal = payload["internal_reference"]

    def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
        return "\n".join(lines)

    risk_rows = [
        {
            "risk": "constraint sign convention mismatch",
            "severity": "high",
            "mitigation": "Inspect pymoo local API and explicitly map repository violations to external convention.",
        },
        {
            "risk": "actual evaluation accounting mismatch",
            "severity": "high",
            "mitigation": "Require exact budget accounting or skip/revise plan before comparison.",
        },
        {
            "risk": "operator family difference",
            "severity": "medium",
            "mitigation": "Record as warning, not fail, and avoid superiority claims.",
        },
        {
            "risk": "dependency missing",
            "severity": "medium",
            "mitigation": "Record dependency_missing/skip artifact rather than failing or installing.",
        },
        {
            "risk": "API version mismatch",
            "severity": "high",
            "mitigation": "Use local import and inspect.signature results as source of truth.",
        },
        {
            "risk": "external parity overclaim",
            "severity": "high",
            "mitigation": "Keep parity claims forbidden until implementation and separate review.",
        },
    ]

    regression_rows = [
        {
            "command": "python -m pytest tests/test_constrained_external_dependency_inspection.py -q",
            "result": "recorded_after_execution",
            "note": "Updated in final task report if run.",
        },
        {
            "command": "python scripts/inspect_constrained_external_dependencies.py --artifact-suffix ext_api_inspect1",
            "result": "PASS",
            "note": "Inspection artifacts generated without optimizer or benchmark execution.",
        },
        {
            "command": "python -m json.tool artifacts/constrained_external_dependency_inspection_results_ext_api_inspect1.json",
            "result": "recorded_after_execution",
            "note": "Updated in final task report if run.",
        },
        {
            "command": "python scripts/check_local_baseline.py --output-dir artifacts/constrained_external_dependency_inspection_guard",
            "result": "recorded_after_execution",
            "note": "Updated in final task report if run.",
        },
    ]

    pymoo_signature_note = "; ".join(
        f"{name}: {value.get('signature')}"
        for name, value in pymoo.get("signatures", {}).items()
    )
    deap_signature_note = "; ".join(
        f"{name}: {value.get('signature')}"
        for name, value in deap.get("signatures", {}).items()
    )

    return "\n".join(
        [
            "# Constrained External Dependency/API Inspection Report",
            "",
            "## 1. Executive Summary",
            "",
            "이번 작업의 목표는 로컬 환경에서 `pymoo`와 `DEAP`의 설치 상태, 버전, import path, constructor signature, constraint 관련 API 리스크, seed/evaluation accounting 리스크를 inspection-only로 확인하는 것이다.",
            "",
            "External comparator는 구현하지 않았고, benchmark와 optimizer는 실행하지 않았으며, dependency도 설치하지 않았다. 추천 comparator는 여전히 `pymoo_constrained_nsga2`이지만, implementation readiness는 local inspection 결과에 따르며 external parity claim은 없다.",
            "",
            f"- pymoo installed/version/status: `{pymoo['installed']}` / `{pymoo.get('version')}` / `{pymoo['recommended_status']}`",
            f"- DEAP installed/version/status: `{deap['installed']}` / `{deap.get('version')}` / `{deap['recommended_status']}`",
            f"- implementation readiness decision: `{payload['inspection_decision']}`",
            "- default 변경 여부: none",
            "- Level 판정 변화 여부: Level 상향 불가; inspection governance 근거만 강화",
            "",
            "## 2. Inspection Scope",
            "",
            table(
                [
                    {"area": "pymoo", "checked": "install/version/import/signature/risk notes", "result": pymoo["recommended_status"]},
                    {"area": "DEAP", "checked": "install/version/import/tool availability/risk notes", "result": deap["recommended_status"]},
                    {"area": "internal pattern", "checked": "external_mo_comparators.py and pyproject extras", "result": "checked"},
                    {"area": "benchmark execution", "checked": "not allowed", "result": "not_executed"},
                    {"area": "optimizer execution", "checked": "not allowed", "result": "not_executed"},
                ],
                ["area", "checked", "result"],
            ),
            "",
            "## 3. pymoo Inspection",
            "",
            table(
                [
                    {"item": "installed/version", "result": f"{pymoo['installed']} / {pymoo.get('version')}", "note": pymoo["import_status"]},
                    {"item": "Problem import", "result": pymoo.get("api_symbols", {}).get("Problem", {}).get("available"), "note": pymoo.get("api_symbols", {}).get("Problem", {}).get("path")},
                    {"item": "NSGA2 import", "result": pymoo.get("api_symbols", {}).get("NSGA2", {}).get("available"), "note": pymoo.get("api_symbols", {}).get("NSGA2", {}).get("path")},
                    {"item": "minimize import", "result": pymoo.get("api_symbols", {}).get("minimize", {}).get("available"), "note": pymoo.get("api_symbols", {}).get("minimize", {}).get("path")},
                    {"item": "constructor signatures", "result": "captured", "note": pymoo_signature_note or "unavailable"},
                    {"item": "constraint API notes", "result": pymoo["constraint_api_notes"].get("constructor_parameter_candidates"), "note": pymoo["constraint_api_notes"].get("constraint_sign_convention")},
                    {"item": "seed/evaluation budget notes", "result": "captured", "note": f"{pymoo['seed_notes'].get('seed_parameter_candidate')}; {pymoo['evaluation_budget_notes'].get('termination_candidate')}"},
                    {"item": "risk summary", "result": len(pymoo.get("risks", [])), "note": "; ".join(pymoo.get("risks", []))},
                    {"item": "recommended status", "result": pymoo["recommended_status"], "note": "planning status only"},
                ],
                ["item", "result", "note"],
            ),
            "",
            "## 4. DEAP Inspection",
            "",
            table(
                [
                    {"item": "installed/version", "result": f"{deap['installed']} / {deap.get('version')}", "note": deap["import_status"]},
                    {"item": "selNSGA2 availability", "result": deap.get("api_symbols", {}).get("selNSGA2", {}).get("available"), "note": deap.get("api_symbols", {}).get("selNSGA2", {}).get("path")},
                    {"item": "selTournamentDCD availability", "result": deap.get("api_symbols", {}).get("selTournamentDCD", {}).get("available"), "note": deap.get("api_symbols", {}).get("selTournamentDCD", {}).get("path")},
                    {"item": "tools.emo availability", "result": deap.get("api_symbols", {}).get("tools.emo", {}).get("available"), "note": deap.get("api_symbols", {}).get("tools.emo", {}).get("path")},
                    {"item": "constraint-domination risk", "result": "not_asserted", "note": deap["constraint_api_notes"].get("inspection_note")},
                    {"item": "signatures", "result": "captured_if_available", "note": deap_signature_note or "unavailable"},
                    {"item": "recommended status", "result": deap["recommended_status"], "note": "secondary comparator only"},
                ],
                ["item", "result", "note"],
            ),
            "",
            "## 5. Internal Optional Dependency Pattern",
            "",
            table(
                [
                    {"item": "external_mo_comparators.py pattern", "result": internal["optional_dependency_pattern"]},
                    {"item": "pyproject extras", "result": internal.get("mo_compare_dependencies")},
                    {"item": "skip artifact convention", "result": internal["existing_skip_pattern"]},
                ],
                ["item", "result"],
            ),
            "",
            "## 6. Implementation Readiness Decision",
            "",
            f"**{payload['inspection_decision']}**",
            "",
            "## 7. Risks",
            "",
            table(risk_rows, ["risk", "severity", "mitigation"]),
            "",
            "## 8. What This Proves",
            "",
            "- local dependency/API status is known.",
            "- missing dependencies are handled as skip/planning status.",
            "- comparator implementation is not yet done.",
            "- benchmark comparison is not yet done.",
            "",
            "## 9. What This Does Not Prove",
            "",
            "- external parity 확보 아님.",
            "- pymoo comparator 구현 완료 아님.",
            "- DEAP comparator 구현 완료 아님.",
            "- default NSGA-II 변경 아님.",
            "- constrained MOEA product readiness 아님.",
            "",
            "## 10. Regression / Governance Check",
            "",
            table(regression_rows, ["command", "result", "note"]),
            "",
            "## 11. Maturity Impact",
            "",
            "**Level 4 근거 강화.** Dependency/API inspection은 algorithm maturity 상향 근거가 아니다. External implementation 전이므로 constrained MOEA maturity 상향은 금지한다. Skip/inspection governance가 명확해졌으므로 실험 툴킷 관점의 Level 4 근거만 강화된다.",
            "",
            "## 12. Recommended Next Work",
            "",
            "Recommended next work: pymoo constrained comparator implementation prompt 작성, 단 `pymoo` 상태가 dependency_missing이면 dependency 설치/optional extra 확인 prompt를 먼저 작성한다. DEAP은 secondary hold로 유지한다.",
            "",
            f"이번 constrained external dependency/API inspection 결과, pymoo 상태는 {pymoo['recommended_status']}이며, DEAP 상태는 {deap['recommended_status']}이고, 다음 단계는 {payload['next_action']}이다.",
            "",
        ]
    )


def build_payload(argv: list[str]) -> dict[str, Any]:
    project_root = _project_root()
    pymoo = _inspect_pymoo()
    deap = _inspect_deap()
    internal = _inspect_internal_reference(project_root)
    decision = _inspection_decision(pymoo, deap)
    if decision not in ALLOWED_INSPECTION_DECISIONS:
        decision = "implementation_not_ready"
    warnings = list(pymoo.get("warnings", [])) + list(deap.get("warnings", []))
    failures: list[str] = []
    return {
        "command": argv,
        "timestamp": datetime.now(UTC).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "benchmark_execution": "not_executed",
        "optimizer_execution": "not_executed",
        "external_comparator_implemented": False,
        "default_nsga2_changed": False,
        "constrained_nsga2_scope_change": "none",
        "dependencies": {"pymoo": pymoo, "deap": deap},
        "internal_reference": internal,
        "inspection_decision": decision,
        "next_action": (
            "pymoo constrained comparator implementation prompt"
            if decision == "pymoo_ready_for_wrapper_planning"
            else "dependency installation/optional extra confirmation prompt"
            if decision == "pymoo_dependency_missing"
            else "manual API review before implementation"
        ),
        "warnings": warnings,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect optional external constrained comparator dependencies without execution.",
    )
    parser.add_argument("--artifact-suffix", default="ext_api_inspect1")
    parser.add_argument("--output-dir", default="artifacts")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = build_payload(sys.argv)
    project_root = _project_root()
    output_dir_input = Path(args.output_dir)
    output_dir = output_dir_input if output_dir_input.is_absolute() else project_root / output_dir_input
    suffix = args.artifact_suffix
    json_path = output_dir / f"constrained_external_dependency_inspection_results_{suffix}.json"
    report_path = output_dir / f"constrained_external_dependency_inspection_report_{suffix}.md"
    payload["artifacts"] = {
        "json": str(json_path),
        "markdown": str(report_path),
    }
    _fresh_path(json_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _fresh_path(report_path).write_text(_render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "inspection_decision": payload["inspection_decision"],
                "artifacts": payload["artifacts"],
                "warnings": payload["warnings"],
                "failures": payload["failures"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
