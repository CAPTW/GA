# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVICES_DIR = PROJECT_ROOT / "services"
if str(SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICES_DIR))

from ga_ops.db import OpsDatabase
from ga_ops.settings import OpsSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or rotate ops API tokens.")
    parser.add_argument("--name", required=True, help="Token name")
    parser.add_argument(
        "--scopes",
        nargs="+",
        default=["ops.read", "ops.write"],
        help="Scopes to attach to the token",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional explicit token value. When omitted a random token is generated.",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Optional override for the ops SQLite database path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = OpsSettings.from_env(project_root=PROJECT_ROOT)
    if args.db_path is not None:
        settings.db_path = Path(args.db_path)
    with OpsDatabase(settings.db_path) as database:
        database.initialize()
        token = database.create_token(args.name, scopes=args.scopes, raw_token=args.token)
        database.log_audit(
            actor="ops_manage_tokens",
            action="create_token",
            resource_type="api_token",
            resource_id=args.name,
            status="success",
            details={"scopes": args.scopes},
        )
    print(json.dumps({"name": args.name, "scopes": args.scopes, "token": token}, indent=2))


if __name__ == "__main__":
    main()
