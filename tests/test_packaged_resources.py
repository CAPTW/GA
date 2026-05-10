from __future__ import annotations

import json

from ga_lab.resources import (
    builtin_resource_names,
    get_builtin_resource,
    materialize_builtin_resource,
    read_builtin_json,
)


def test_builtin_resource_catalog_contains_expected_names() -> None:
    assert "onemax_small" in builtin_resource_names("preset")
    assert "tsp_medium_hybrid" in builtin_resource_names("preset")
    assert "onemax_pure_ga_demo" in builtin_resource_names("demo")
    assert "baseline_onemax_demo_manifest" in builtin_resource_names("demo-manifest")


def test_builtin_resource_loader_reads_json() -> None:
    preset_payload = read_builtin_json("preset", "onemax_small")
    demo_payload = read_builtin_json("demo", "zdt1_nsga2_demo")

    assert preset_payload["problem"] == "onemax"
    assert preset_payload["run_name"] == "onemax_small"
    assert demo_payload["algorithm"] == "nsga2"
    assert demo_payload["problem"] == "zdt1"


def test_materialize_builtin_resource_writes_real_file(tmp_path) -> None:
    output_path = materialize_builtin_resource(
        "preset",
        "onemax_small",
        tmp_path / "presets" / "onemax_small.json",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path.exists()
    assert payload["run_name"] == "onemax_small"


def test_builtin_resource_repo_mapping_is_stable() -> None:
    resource = get_builtin_resource("preset", "zdt1_large")
    assert resource.repo_relative_path == "configs/presets/zdt1_large.json"
    assert resource.resource_uri == "preset:zdt1_large"
