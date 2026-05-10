# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ga_lab.local_candidate import evaluate_candidate_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a local candidate experiment against the frozen baseline snapshot."
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Candidate manifest path, for example configs/local_candidates/example_zdt1_candidate.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/local_candidates",
        help="Directory where candidate reports will be written.",
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Do not run a candidate study; use existing output only when available.",
    )
    parser.add_argument(
        "--use-existing-output",
        action="store_true",
        help="Require and reuse the manifest's existing_output_dir instead of running a study.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_candidate_manifest(
        args.candidate,
        project_root=PROJECT_ROOT,
        output_root=args.output_root,
        no_execute=args.no_execute,
        use_existing_output=args.use_existing_output,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
