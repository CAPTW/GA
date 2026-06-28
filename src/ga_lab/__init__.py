"""Evidence-backed GA and solver-choice research harness."""

from importlib.metadata import PackageNotFoundError, version

from ga_lab.config import GAConfig
from ga_lab.runner import run_experiment

try:
    __version__ = version("evolutionary-solver-benchmark-lab")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["GAConfig", "run_experiment", "__version__"]
