from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REGISTRY_SCHEMA_VERSION = 1
ALLOWED_STATUSES = {"official", "family_conditional", "experimental", "not_recommended"}
ALLOWED_SCOPES = {"internal", "external", "family_conditional"}
ALLOWED_GOVERNANCE_MODES = {"ci_gated", "report_only"}
REQUIRED_CLAIM_FIELDS = {
    "claim_id",
    "label",
    "status",
    "evidence_scope",
    "problem_family",
    "validated_ranges",
    "comparators",
    "metrics",
    "pass_condition",
    "warning_threshold",
    "source_summary_paths",
    "doc_locations",
    "notes",
    "governance",
}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_claim_registry(payload: dict[str, Any], *, registry_path: Path | None = None) -> None:
    _assert(isinstance(payload, dict), "Claim registry must be a JSON object.")
    _assert(
        payload.get("registry_schema_version") == REGISTRY_SCHEMA_VERSION,
        f"claim_registry registry_schema_version must be {REGISTRY_SCHEMA_VERSION}.",
    )
    claims = payload.get("claims")
    _assert(isinstance(claims, list) and claims, "Claim registry requires a non-empty claims list.")
    root = registry_path.resolve().parents[1] if registry_path is not None else None
    seen_ids: set[str] = set()

    for claim in claims:
        _assert(isinstance(claim, dict), "Each claim registry entry must be a JSON object.")
        missing = sorted(REQUIRED_CLAIM_FIELDS.difference(claim))
        _assert(
            not missing,
            (
                f"Claim entry {claim.get('claim_id', '<missing-id>')} "
                f"is missing fields: {', '.join(missing)}"
            ),
        )

        claim_id = claim["claim_id"]
        _assert(isinstance(claim_id, str) and claim_id, "claim_id must be a non-empty string.")
        _assert(claim_id not in seen_ids, f"Duplicate claim_id detected: {claim_id}")
        seen_ids.add(claim_id)

        _assert(
            claim["status"] in ALLOWED_STATUSES,
            f"{claim_id}: status must be one of {sorted(ALLOWED_STATUSES)}",
        )
        _assert(
            claim["evidence_scope"] in ALLOWED_SCOPES,
            f"{claim_id}: evidence_scope must be one of {sorted(ALLOWED_SCOPES)}",
        )
        _assert(
            isinstance(claim["validated_ranges"], list) and claim["validated_ranges"],
            f"{claim_id}: validated_ranges must be a non-empty list.",
        )
        _assert(
            isinstance(claim["comparators"], list) and claim["comparators"],
            f"{claim_id}: comparators must be a non-empty list.",
        )
        _assert(
            isinstance(claim["metrics"], list) and claim["metrics"],
            f"{claim_id}: metrics must be a non-empty list.",
        )
        _assert(
            isinstance(claim["source_summary_paths"], list) and claim["source_summary_paths"],
            f"{claim_id}: source_summary_paths must be a non-empty list.",
        )
        _assert(
            isinstance(claim["doc_locations"], list) and claim["doc_locations"],
            f"{claim_id}: doc_locations must be a non-empty list.",
        )

        governance = claim["governance"]
        _assert(isinstance(governance, dict), f"{claim_id}: governance must be an object.")
        _assert(
            governance.get("mode") in ALLOWED_GOVERNANCE_MODES,
            f"{claim_id}: governance.mode must be one of {sorted(ALLOWED_GOVERNANCE_MODES)}",
        )

        checks = claim.get("checks", [])
        _assert(isinstance(checks, list), f"{claim_id}: checks must be a list when provided.")
        for index, check in enumerate(checks, start=1):
            _assert(isinstance(check, dict), f"{claim_id}: check #{index} must be an object.")
            for field in ("summary_stem", "rowset", "match", "pass_if"):
                _assert(
                    field in check,
                    f"{claim_id}: check #{index} missing required field `{field}`.",
                )
            _assert(
                isinstance(check["match"], dict),
                f"{claim_id}: check #{index} match must be an object.",
            )
            _assert(
                isinstance(check["pass_if"], list) and check["pass_if"],
                f"{claim_id}: check #{index} pass_if must be a non-empty list.",
            )
            warn_if = check.get("warn_if", [])
            _assert(
                isinstance(warn_if, list),
                f"{claim_id}: check #{index} warn_if must be a list.",
            )

        if root is not None:
            for doc_path in claim["doc_locations"]:
                _assert(
                    (root / doc_path).exists(),
                    f"{claim_id}: referenced doc path does not exist: {doc_path}",
                )


def load_claim_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path).resolve()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_claim_registry(payload, registry_path=registry_path)
    return payload
