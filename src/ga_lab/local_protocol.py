from __future__ import annotations

import csv
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ga_lab.local_experiments import run_local_study
from ga_lab.project_paths import find_project_root

_MATRIX_RELATIVE_PATH = Path("configs") / "local_protocols" / "local_operating_protocols.json"
_LOCAL_STUDY_RELATIVE_DIR = Path("configs") / "local_studies"


def _project_root() -> Path:
    root = find_project_root(Path(__file__).resolve())
    if root is None:
        raise RuntimeError("Could not locate the GA project root")
    return root


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in value.strip().lower())
    safe = safe.strip("_")
    return safe or "value"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _matrix_path(matrix_ref: str | Path | None = None) -> Path:
    if matrix_ref is None:
        return (_project_root() / _MATRIX_RELATIVE_PATH).resolve()
    candidate = Path(matrix_ref)
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.exists():
        return candidate.resolve()
    return (_project_root() / candidate).resolve()


def _study_path(study_ref: str) -> Path:
    name = study_ref if study_ref.endswith(".json") else f"{study_ref}.json"
    path = (_project_root() / _LOCAL_STUDY_RELATIVE_DIR / name).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Local study not found: {study_ref}")
    return path


def load_local_protocol_matrix(matrix_ref: str | Path | None = None) -> dict[str, Any]:
    path = _matrix_path(matrix_ref)
    payload = _load_json(path)
    protocols = payload.get("protocols")
    if not isinstance(protocols, dict) or not protocols:
        raise ValueError("Protocol matrix must contain a non-empty 'protocols' object")
    return payload


def _resolve_profile_value(default_profiles: dict[str, Any], value: Any) -> Any:
    if isinstance(value, str):
        if value in default_profiles:
            return _resolve_profile_value(default_profiles, default_profiles[value])
        return value
    if isinstance(value, dict):
        return {
            key: _resolve_profile_value(default_profiles, nested)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_resolve_profile_value(default_profiles, nested) for nested in value]
    return value


def protocol_matrix_rows(
    matrix: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    payload = matrix or load_local_protocol_matrix()
    protocols = payload["protocols"]
    rows: list[dict[str, str]] = []
    for problem in ("tsp", "zdt1", "knapsack", "onemax"):
        entry = protocols.get(problem)
        if not isinstance(entry, dict):
            continue
        modes = entry.get("modes", {})
        default_profiles = entry.get("default_profiles", {})
        explore_rule = str(modes.get("explore", {}).get("rule", "n/a"))
        compare_rule = str(
            modes.get("compare", modes.get("sanity", modes.get("control", {}))).get("rule", "n/a")
        )
        final_rule = str(modes.get("final", modes.get("sanity", modes.get("control", {}))).get("rule", "n/a"))
        if problem == "knapsack":
            final_rule = "Broad default parked; use greedy by default and keep repair_only as a narrow sanity note."
        if problem == "onemax":
            compare_rule = "Control-only; no meaningful compare ladder."
            final_rule = "Control-only; no final escalation path."
        seed_ladders = []
        for mode_name in ("explore", "compare", "final", "sanity", "control"):
            mode_entry = modes.get(mode_name)
            if isinstance(mode_entry, dict):
                ladder = mode_entry.get("seed_ladder")
                if isinstance(ladder, list) and ladder:
                    seed_ladders.append(f"{mode_name}:{'/'.join(str(int(v)) for v in ladder)}")
        stop_semantics = []
        for mode_name in ("explore", "compare", "final", "sanity", "control"):
            mode_entry = modes.get(mode_name)
            if isinstance(mode_entry, dict):
                text = mode_entry.get("stop_label_semantics")
                if isinstance(text, str) and text:
                    stop_semantics.append(f"{mode_name}: {text}")
        rows.append(
            {
                "problem": problem,
                "exploratory_mode_rule": explore_rule,
                "compare_mode_rule": compare_rule,
                "final_mode_rule": final_rule,
                "default_profiles": json.dumps(
                    _resolve_profile_value(default_profiles, default_profiles),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "seed_ladder": "; ".join(seed_ladders) if seed_ladders else "n/a",
                "stop_label_semantics": " | ".join(stop_semantics),
            }
        )
    return rows


def _study_seed_ids(study_ref: str | None, *, count: int) -> list[int]:
    if study_ref is None or count <= 0:
        return []
    payload = _load_json(_study_path(study_ref))
    raw_seeds = payload.get("seeds", [])
    seed_ids = [
        int(value)
        for value in raw_seeds
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    return seed_ids[:count]


def resolve_local_protocol(
    *,
    problem: str,
    mode: str,
    case_group: str | None = None,
    quality_sensitive: bool = False,
    anti_case_suspected: bool = False,
    final_safety: bool = False,
    borderline: bool = False,
    seed_count: int | None = None,
    matrix_ref: str | Path | None = None,
) -> dict[str, Any]:
    matrix = load_local_protocol_matrix(matrix_ref)
    protocols = matrix["protocols"]
    if problem not in protocols:
        available = ", ".join(sorted(protocols))
        raise ValueError(f"Unknown problem '{problem}'. Available: {available}")

    problem_entry = deepcopy(protocols[problem])
    modes = problem_entry.get("modes", {})
    if mode not in modes:
        available = ", ".join(sorted(modes))
        raise ValueError(f"Unsupported mode '{mode}' for problem '{problem}'. Available: {available}")

    requested_mode = mode
    effective_mode = mode
    case_group_key = (case_group or "overall").strip() or "overall"
    mode_entry = deepcopy(modes[mode])
    notes: list[str] = []

    if problem == "tsp" and mode == "compare":
        if quality_sensitive or anti_case_suspected or case_group_key == "anti_case":
            effective_mode = str(mode_entry.get("quality_sensitive_direct_mode", "final"))
            mode_entry = deepcopy(modes[effective_mode])
            notes.append(
                "Anti-case suspicion or quality-sensitive intent bypasses paired compare and goes straight to the Q final path."
            )
        else:
            override = mode_entry.get("case_group_overrides", {}).get(case_group_key)
            if isinstance(override, dict):
                if "direct_mode" in override:
                    effective_mode = str(override["direct_mode"])
                    mode_entry = deepcopy(modes[effective_mode])
                else:
                    if "initial_seed_count" in override:
                        mode_entry["initial_seed_count"] = int(override["initial_seed_count"])
                    if "escalation_path" in override:
                        mode_entry["escalation_path"] = list(override["escalation_path"])
                note = override.get("note")
                if isinstance(note, str) and note:
                    notes.append(note)

    if problem == "zdt1" and mode == "compare" and final_safety:
        effective_mode = str(mode_entry.get("final_safety_direct_mode", "final"))
        mode_entry = deepcopy(modes[effective_mode])
        notes.append("Final safety intent bypasses the hopeful fast ladder and goes straight to Q.")

    if problem == "knapsack" and mode == "sanity":
        override = None
        if borderline:
            mode_entry["initial_seed_count"] = int(mode_entry.get("borderline_seed_count", 5))
            mode_entry["escalation_path"] = []
            notes.append("Borderline family hint enabled the longer 5-seed sanity check.")
        elif case_group_key != "overall":
            override = mode_entry.get("case_group_overrides", {}).get(case_group_key)
            if isinstance(override, dict):
                if "initial_seed_count" in override:
                    mode_entry["initial_seed_count"] = int(override["initial_seed_count"])
                if "escalation_path" in override:
                    mode_entry["escalation_path"] = list(override["escalation_path"])
                note = override.get("note")
                if isinstance(note, str) and note:
                    notes.append(note)

    default_profiles = problem_entry.get("default_profiles", {})
    recommended_profile = _resolve_profile_value(
        default_profiles, mode_entry.get("recommended_profile")
    )

    initial_seed_count = int(mode_entry.get("initial_seed_count", 1))
    if seed_count is not None:
        initial_seed_count = int(seed_count)
        notes.append("Seed count was overridden manually for this protocol read.")

    study_refs = problem_entry.get("study_refs", {})
    study_ref: str | None = None
    if problem in {"tsp", "zdt1"}:
        study_ref = str(study_refs.get("compare", "")) or None
    elif problem == "knapsack":
        study_ref = str(study_refs.get("sanity", "")) or None
    elif problem == "onemax":
        study_ref = str(study_refs.get("control", "")) or None

    seed_ids = _study_seed_ids(study_ref, count=initial_seed_count)
    paired_compare_needed = bool(mode_entry.get("paired_compare_needed", False))
    if requested_mode == "compare" and effective_mode == "final":
        paired_compare_needed = False

    public_payload: dict[str, Any] = {
        "problem": problem,
        "requested_mode": requested_mode,
        "mode": effective_mode,
        "case_group": case_group_key,
        "mode_rule": str(mode_entry.get("rule", "")),
        "recommended_profile": recommended_profile,
        "default_profiles": _resolve_profile_value(default_profiles, default_profiles),
        "paired_compare_needed": paired_compare_needed,
        "initial_seed_count": initial_seed_count,
        "seed_ladder": list(mode_entry.get("seed_ladder", [])),
        "seed_ids": seed_ids,
        "escalation_path": list(mode_entry.get("escalation_path", [])),
        "final_confirm_recommendation": str(mode_entry.get("final_confirm_recommendation", "")),
        "stop_label": str(mode_entry.get("stop_label", "")),
        "stop_label_semantics": str(mode_entry.get("stop_label_semantics", "")),
        "rationale": " ".join(
            part for part in [str(mode_entry.get("rationale", "")), *notes] if part
        ).strip(),
        "protocol_source_study": study_ref,
    }

    if requested_mode == "compare" and effective_mode == "compare":
        if problem == "tsp":
            public_payload["study_mode"] = str(mode_entry.get("study_mode", "exploratory"))
            if case_group_key == "rescue_target":
                public_payload["study_scope"] = "case_group"
                public_payload["study_case_group"] = "rescue_target"
            else:
                public_payload["study_scope"] = "overall"
                public_payload["study_case_group"] = "overall"
        elif problem == "zdt1":
            public_payload["study_mode"] = str(mode_entry.get("study_mode", "exploratory"))
            public_payload["study_scope"] = "overall"
            public_payload["study_case_group"] = "overall"

    if requested_mode == "sanity":
        public_payload["study_mode"] = str(mode_entry.get("study_mode", "repair_note"))
        if case_group_key == "overall" and not borderline:
            public_payload["study_scope"] = "overall"
            public_payload["study_case_group"] = "overall"
        else:
            public_payload["study_scope"] = "case_group"
            public_payload["study_case_group"] = "tight_capacity_small" if borderline else case_group_key

    if requested_mode == "control":
        public_payload["study_mode"] = "control"
        public_payload["study_scope"] = "overall"
        public_payload["study_case_group"] = "overall"

    if isinstance(mode_entry.get("variant_key"), str):
        public_payload["variant_key"] = mode_entry["variant_key"]
    if isinstance(mode_entry.get("quality_variant_key"), str):
        public_payload["quality_variant_key"] = mode_entry["quality_variant_key"]
    if isinstance(mode_entry.get("fast_variant_key"), str):
        public_payload["fast_variant_key"] = mode_entry["fast_variant_key"]
    return public_payload


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _prepare_study_payload(
    *,
    study_ref: str,
    study_name: str,
    seed_count: int,
    variant_names: list[str],
    case_group: str | None,
) -> dict[str, Any]:
    payload = _load_json(_study_path(study_ref))
    payload["study_name"] = study_name
    payload["description"] = f"{payload.get('description', '').strip()} Protocol helper execution."
    payload["seeds"] = list(payload.get("seeds", []))[:seed_count]

    if case_group is not None and "cases" in payload:
        filtered_cases = [
            case
            for case in payload.get("cases", [])
            if isinstance(case, dict) and str(case.get("group") or "overall") == case_group
        ]
        if not filtered_cases:
            raise ValueError(f"No cases matched case_group='{case_group}' in study '{study_ref}'")
        payload["cases"] = filtered_cases

    if "variant_overrides" in payload:
        payload["variant_overrides"] = {
            key: value
            for key, value in payload["variant_overrides"].items()
            if key in variant_names
        }
        if not payload["variant_overrides"]:
            raise ValueError(f"No variants matched {variant_names} in study '{study_ref}'")
    if "sweep" in payload and isinstance(payload["sweep"], dict):
        if "study_variant" in payload["sweep"]:
            payload["sweep"]["study_variant"] = [
                value
                for value in payload["sweep"]["study_variant"]
                if value in variant_names
            ]
            if not payload["sweep"]["study_variant"]:
                raise ValueError(f"No sweep variants matched {variant_names} in study '{study_ref}'")
    return payload


def _variant_names_for_execution(decision: dict[str, Any]) -> list[str]:
    requested_mode = str(decision.get("requested_mode"))
    effective_mode = str(decision.get("mode"))
    problem = str(decision.get("problem"))

    if requested_mode == "compare" and effective_mode == "compare":
        quality_key = decision.get("quality_variant_key")
        fast_key = decision.get("fast_variant_key")
        return [
            value
            for value in (quality_key, fast_key)
            if isinstance(value, str) and value
        ]
    if requested_mode == "sanity" and problem == "knapsack":
        return ["greedy_local_search", "none", "repair_only"]
    if requested_mode == "control" and problem == "onemax":
        return ["none", "early_stop_reference"]
    variant_key = decision.get("variant_key")
    if isinstance(variant_key, str) and variant_key:
        return [variant_key]
    return []


def _execution_case_group(decision: dict[str, Any]) -> str | None:
    requested_mode = str(decision.get("requested_mode"))
    case_group = str(decision.get("case_group") or "overall")
    if requested_mode in {"compare", "sanity"} and case_group != "overall":
        return case_group
    if requested_mode == "final" and case_group != "overall":
        return case_group
    return None


def _select_sequential_row(
    rows: list[dict[str, str]],
    *,
    mode: str,
    scope: str,
    case_group: str,
    seed_count: int,
) -> dict[str, str] | None:
    for row in rows:
        if row.get("mode") != mode:
            continue
        if row.get("scope") != scope:
            continue
        if row.get("case_group") != case_group:
            continue
        try:
            row_seed = int(float(row.get("seed_count", "0")))
        except ValueError:
            continue
        if row_seed == seed_count:
            return row
    return None


def _map_protocol_label(problem: str, requested_mode: str, raw_label: str | None) -> str | None:
    if raw_label is None:
        return None
    if problem == "knapsack" and requested_mode == "sanity":
        return "repair_note_stable" if raw_label == "accept_fast_exploratory" else raw_label
    if problem == "onemax" and requested_mode == "control":
        return "control_stable"
    return raw_label


def _execute_protocol_study(
    decision: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    study_ref = decision.get("protocol_source_study")
    if not isinstance(study_ref, str) or not study_ref:
        return {}

    variant_names = _variant_names_for_execution(decision)
    if not variant_names:
        return {}

    case_group = _execution_case_group(decision)
    payload = _prepare_study_payload(
        study_ref=study_ref,
        study_name=f"protocol_{decision['problem']}_{decision['mode']}",
        seed_count=int(decision["initial_seed_count"]),
        variant_names=variant_names,
        case_group=case_group,
    )
    manifest_path = output_dir / "protocol_study.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    study_result = run_local_study(manifest_path, output_root=output_dir / "study_outputs")
    output_paths = dict(study_result)
    output_paths["protocol_study_manifest"] = str(manifest_path.resolve())

    sequential_path = study_result.get("sequential_decision_table_csv")
    requested_mode = str(decision.get("requested_mode"))
    if (
        isinstance(sequential_path, str)
        and sequential_path
        and requested_mode in {"compare", "sanity", "control"}
    ):
        rows = _load_csv_rows(Path(sequential_path))
        selected = _select_sequential_row(
            rows,
            mode=str(decision.get("study_mode", "")),
            scope=str(decision.get("study_scope", "overall")),
            case_group=str(decision.get("study_case_group", "overall")),
            seed_count=int(decision["initial_seed_count"]),
        )
        if selected is not None:
            output_paths["selected_sequential_row"] = selected
    return output_paths


def _render_markdown(decision: dict[str, Any]) -> str:
    lines = [
        f"# Local Protocol Decision: {decision['problem']} / {decision['requested_mode']}",
        "",
        f"- Effective mode: `{decision['mode']}`",
        f"- Case group: `{decision['case_group']}`",
        f"- Recommended profile: `{json.dumps(decision['recommended_profile'], ensure_ascii=False)}`",
        f"- Paired compare needed: `{str(bool(decision['paired_compare_needed'])).lower()}`",
        f"- Initial seed count: `{decision['initial_seed_count']}`",
        f"- Escalation path: `{decision['escalation_path']}`",
        f"- Final confirm recommendation: {decision['final_confirm_recommendation']}",
        f"- Stop label: `{decision.get('decision_label') or decision['stop_label']}`",
        f"- Stop-label semantics: {decision['stop_label_semantics']}",
        "",
        "## Rationale",
        "",
        decision["rationale"] or "Use the frozen local rule as-is.",
    ]
    output_paths = decision.get("output_paths") or {}
    if isinstance(output_paths, dict) and output_paths:
        lines.extend(["", "## Output Paths", ""])
        for key, value in output_paths.items():
            if key == "selected_sequential_row":
                continue
            lines.append(f"- {key}: `{value}`")
        selected_row = output_paths.get("selected_sequential_row")
        if isinstance(selected_row, dict):
            lines.extend(
                [
                    "",
                    "## Selected Sequential Row",
                    "",
                    f"- study_decision_label: `{selected_row.get('decision_label', '')}`",
                    f"- stage_action: `{selected_row.get('stage_action', '')}`",
                    f"- seed_count: `{selected_row.get('seed_count', '')}`",
                    f"- actual_evaluations_used: `{selected_row.get('actual_evaluations_used', '')}`",
                    f"- actual_eval_savings_vs_full_pct: `{selected_row.get('actual_eval_savings_vs_full_pct', '')}`",
                ]
            )
    return "\n".join(lines).strip() + "\n"


def run_local_protocol(
    *,
    problem: str,
    mode: str,
    output_root: str | Path = "outputs/local_protocols",
    case_group: str | None = None,
    quality_sensitive: bool = False,
    anti_case_suspected: bool = False,
    final_safety: bool = False,
    borderline: bool = False,
    seed_count: int | None = None,
    execute: bool = False,
    explain: bool = False,
    matrix_ref: str | Path | None = None,
) -> dict[str, Any]:
    decision = resolve_local_protocol(
        problem=problem,
        mode=mode,
        case_group=case_group,
        quality_sensitive=quality_sensitive,
        anti_case_suspected=anti_case_suspected,
        final_safety=final_safety,
        borderline=borderline,
        seed_count=seed_count,
        matrix_ref=matrix_ref,
    )
    if not explain:
        decision["rationale"] = decision["rationale"].split(". ")[0].strip()

    output_dir = Path(output_root) / (
        f"{_timestamp()}_{_safe_name(problem)}_{_safe_name(mode)}"
        + (f"_{_safe_name(str(case_group))}" if case_group else "")
    )
    if not output_dir.is_absolute():
        output_dir = (_project_root() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, Any] = {}
    if execute:
        output_paths = _execute_protocol_study(decision, output_dir=output_dir)
        selected_row = output_paths.get("selected_sequential_row")
        if isinstance(selected_row, dict):
            study_label = selected_row.get("decision_label")
            decision["study_decision_label"] = study_label
            decision["decision_label"] = _map_protocol_label(
                str(decision["problem"]),
                str(decision["requested_mode"]),
                str(study_label) if isinstance(study_label, str) else None,
            )
            decision["stage_action"] = selected_row.get("stage_action")
            decision["actual_evaluations_used"] = selected_row.get("actual_evaluations_used")
            decision["actual_eval_savings_vs_full_pct"] = selected_row.get(
                "actual_eval_savings_vs_full_pct"
            )
        else:
            decision["decision_label"] = decision["stop_label"]
    else:
        decision["decision_label"] = decision["stop_label"]

    decision["output_paths"] = output_paths

    json_path = output_dir / "protocol_decision.json"
    md_path = output_dir / "protocol_decision.md"
    public_payload = {
        key: value
        for key, value in decision.items()
        if key
        not in {
            "protocol_source_study",
            "study_mode",
            "study_scope",
            "study_case_group",
            "variant_key",
            "quality_variant_key",
            "fast_variant_key",
        }
    }
    json_path.write_text(json.dumps(public_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_render_markdown(public_payload), encoding="utf-8")
    public_payload["protocol_decision_json"] = str(json_path.resolve())
    public_payload["protocol_decision_md"] = str(md_path.resolve())
    return public_payload
