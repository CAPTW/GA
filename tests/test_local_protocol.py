from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ga_lab.local_protocol import load_local_protocol_matrix, protocol_matrix_rows


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_json_command(*args: str, cwd: Path | None = None) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, *args],
        cwd=cwd or _project_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def test_protocol_matrix_load() -> None:
    matrix = load_local_protocol_matrix()
    rows = protocol_matrix_rows(matrix)

    assert matrix["protocols"]["tsp"]["modes"]["compare"]["initial_seed_count"] == 3
    assert matrix["protocols"]["zdt1"]["modes"]["final"]["initial_seed_count"] == 8
    assert matrix["protocols"]["knapsack"]["modes"]["sanity"]["initial_seed_count"] == 3
    assert matrix["protocols"]["onemax"]["modes"]["control"]["initial_seed_count"] == 1
    assert any(row["problem"] == "tsp" for row in rows)
    assert any(row["problem"] == "zdt1" for row in rows)
    assert any(row["problem"] == "knapsack" for row in rows)
    assert any(row["problem"] == "onemax" for row in rows)


def test_tsp_protocol_runner_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "protocols"

    explore = _run_json_command(
        "scripts/run_local_protocol.py",
        "--problem",
        "tsp",
        "--mode",
        "explore",
        "--output-root",
        str(output_root),
    )
    compare = _run_json_command(
        "scripts/run_local_protocol.py",
        "--problem",
        "tsp",
        "--mode",
        "compare",
        "--case-group",
        "rescue_target",
        "--output-root",
        str(output_root),
    )
    final = _run_json_command(
        "scripts/run_local_protocol.py",
        "--problem",
        "tsp",
        "--mode",
        "final",
        "--anti-case-suspected",
        "--output-root",
        str(output_root),
    )

    assert str(explore["recommended_profile"]).endswith("tsp_seeded_swap_local_fast.json")
    assert explore["initial_seed_count"] == 3
    assert Path(str(explore["protocol_decision_json"])).exists()

    compare_profile = compare["recommended_profile"]
    assert isinstance(compare_profile, dict)
    assert compare["paired_compare_needed"] is True
    assert compare["initial_seed_count"] == 5
    assert Path(str(compare["protocol_decision_md"])).exists()

    assert str(final["recommended_profile"]).endswith("tsp_seeded_swap_local.json")
    assert final["mode"] == "final"
    assert final["initial_seed_count"] == 8


def test_zdt1_protocol_runner_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "protocols"

    explore = _run_json_command(
        "scripts/run_local_protocol.py",
        "--problem",
        "zdt1",
        "--mode",
        "explore",
        "--output-root",
        str(output_root),
    )
    final = _run_json_command(
        "scripts/run_local_protocol.py",
        "--problem",
        "zdt1",
        "--mode",
        "final",
        "--output-root",
        str(output_root),
    )

    assert str(explore["recommended_profile"]).endswith("zdt1_diversity_injection_fast.json")
    assert explore["initial_seed_count"] == 3
    assert str(final["recommended_profile"]).endswith("zdt1_diversity_injection.json")
    assert final["initial_seed_count"] == 8


def test_knapsack_protocol_runner_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "protocols"
    payload = _run_json_command(
        "scripts/run_local_protocol.py",
        "--problem",
        "knapsack",
        "--mode",
        "sanity",
        "--borderline",
        "--output-root",
        str(output_root),
    )

    assert payload["mode"] == "sanity"
    assert payload["initial_seed_count"] == 5
    assert payload["stop_label"] == "repair_note_stable"
    assert Path(str(payload["protocol_decision_json"])).exists()


def test_protocol_runner_execute_compare_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "protocols"
    payload = _run_json_command(
        "scripts/run_local_protocol.py",
        "--problem",
        "zdt1",
        "--mode",
        "compare",
        "--execute",
        "--output-root",
        str(output_root),
    )

    assert payload["paired_compare_needed"] is True
    assert payload["decision_label"] == "accept_fast_exploratory"
    output_paths = payload["output_paths"]
    assert isinstance(output_paths, dict)
    assert Path(str(output_paths["study_dir"])).exists()
    assert Path(str(output_paths["sequential_decision_table_csv"])).exists()


def test_local_protocol_docs_reference_real_commands() -> None:
    project_root = _project_root()
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    guide = (project_root / "docs" / "local_experiment_guide.md").read_text(encoding="utf-8")
    protocol_guide = (project_root / "docs" / "local_protocol_guide.md").read_text(encoding="utf-8")
    examples = (project_root / "examples" / "README.md").read_text(encoding="utf-8")

    assert "python scripts/run_local_protocol.py --problem tsp --mode explore" in readme
    assert "python scripts/run_local_protocol.py --problem tsp --mode compare --case-group rescue_target" in readme
    assert "python scripts/run_local_protocol.py --problem tsp --mode final --anti-case-suspected" in readme
    assert "[Local protocol guide](docs/local_protocol_guide.md)" in readme

    assert "python scripts/run_local_protocol.py --problem tsp --mode explore" in guide
    assert "python scripts/run_local_protocol.py --problem zdt1 --mode compare --final-safety" in guide
    assert "[Local protocol guide](local_protocol_guide.md)" in guide

    assert "python scripts/run_local_protocol.py --problem knapsack --mode sanity --borderline" in protocol_guide
    assert "python scripts/run_local_protocol.py --problem onemax --mode control" in protocol_guide

    assert "python scripts/run_local_protocol.py --problem tsp --mode explore" in examples
    assert "python scripts/run_local_protocol.py --problem zdt1 --mode compare --final-safety" in examples
