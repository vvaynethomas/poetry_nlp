"""Paths and shared settings. Everything is relative to the repository root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "kaggle_poem_dataset.csv"
INTERIM_DIR = DATA_DIR / "interim"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

POEMS_PKL = INTERIM_DIR / "poems.pkl"
LINES_PKL = INTERIM_DIR / "lines.pkl"
POEM_VECTORS = INTERIM_DIR / "poem_vectors.npy"
DOC2VEC_MODEL = MODELS_DIR / "poem_doc2vec_dbow.model"

KAGGLE_DATASET = "johnhallman/complete-poetryfoundationorg-dataset"

SEED = 13


def ensure_dirs():
    for d in (INTERIM_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, RAW_CSV.parent):
        d.mkdir(parents=True, exist_ok=True)
