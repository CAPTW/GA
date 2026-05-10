PYTHON ?= python
RUFF_PATHS := src tests scripts services

.PHONY: test run grid summary sweep compare prune dashboard ops-dashboard ops-sync ops-api ops-scheduler rc-regression weekly-report lint lint-fix format typecheck benchmark-smoke ci-check mlflow-ui track-mlflow track-wandb-offline nightly nightly-dry-run

test:
	$(PYTHON) -m pytest

run:
	$(PYTHON) scripts/run_experiment.py --config configs/onemax_baseline.json

run-real:
	$(PYTHON) scripts/run_experiment.py --config configs/onemax_real_ga.json

grid:
	$(PYTHON) scripts/run_grid.py --config configs/onemax_baseline.json --seeds 5

sweep:
	$(PYTHON) scripts/run_baselines.py --manifest configs/comparisons/onemax_operator_compare_10seeds.json

compare:
	$(PYTHON) scripts/run_baselines.py --manifest configs/comparisons/onemax_operator_compare_10seeds.json

summary:
	$(PYTHON) scripts/summarize_results.py --results-dir outputs

prune:
	$(PYTHON) scripts/prune_results.py --results-dir outputs

dashboard:
	$(PYTHON) -m streamlit run scripts/streamlit_dashboard.py

ops-dashboard:
	$(PYTHON) -m streamlit run scripts/ops_dashboard.py

ops-sync:
	$(PYTHON) scripts/ops_sync_results.py --results-dir outputs

ops-api:
	$(PYTHON) -m uvicorn ga_ops.app:app --host 0.0.0.0 --port 8000 --app-dir services

ops-scheduler:
	$(PYTHON) scripts/run_ops_scheduler.py

rc-regression:
	$(PYTHON) scripts/run_release_candidate_regression.py

weekly-report:
	$(PYTHON) scripts/ops_generate_weekly_report.py

lint:
	$(PYTHON) -m ruff check $(RUFF_PATHS)

lint-fix:
	$(PYTHON) -m ruff check $(RUFF_PATHS) --fix

format:
	$(PYTHON) -m ruff format $(RUFF_PATHS)

typecheck:
	$(PYTHON) -m mypy src

benchmark-smoke:
	$(PYTHON) scripts/run_baselines.py --manifest configs/ci/baseline_smoke.json --output-root outputs_test

ci-check:
	$(PYTHON) -m ruff check $(RUFF_PATHS)
	$(PYTHON) -m ruff format --check $(RUFF_PATHS)
	$(PYTHON) -m mypy src
	$(PYTHON) -m pytest
	$(PYTHON) scripts/run_baselines.py --manifest configs/ci/baseline_smoke.json --output-root outputs_test

nightly:
	$(PYTHON) scripts/run_nightly.py

nightly-dry-run:
	$(PYTHON) scripts/run_nightly.py --dry-run

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri ./mlruns --port 5000

track-mlflow:
	$(PYTHON) scripts/run_experiment.py --config configs/tracking/onemax_mlflow_local.json

track-wandb-offline:
	$(PYTHON) scripts/run_experiment.py --config configs/tracking/onemax_wandb_offline.json
