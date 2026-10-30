"""Paths and default settings shared by every stage."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("NSCA_DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.environ.get("NSCA_MODEL_DIR", ROOT / "models"))
REPORT_DIR = ROOT / "reports"
HPO_OBO = DATA_DIR / "hpo" / "hp.obo"

DATASETS = ("mendeley", "ddxplus", "phenopackets")
DEFAULT_DATASET = os.environ.get("NSCA_DEFAULT_DATASET", "ddxplus")
SEED = 42
