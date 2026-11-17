"""Smoke tests on a tiny synthetic corpus (no Kaggle data needed)."""
import random

import numpy as np
import pandas as pd
import pytest

from poetry_nlp import config, prepare

WORDS = ("river stone light winter field morning bird window salt rain mother city "
         "night garden bone ocean letter fire road snow").split()


def synthetic_csv(path, n=60, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        lines = [" ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 8)))
                 for _ in range(rng.randint(4, 12))]
        rows.append({"Author": f"Author {i % 9}", "Title": f"Title {i}",
                     "Poetry Foundation ID": i, "Content": "\n".join(lines)})
    pd.DataFrame(rows).to_csv(path)


def test_clean_text():
    assert prepare.clean_text("Hello,  World! <b>2021</b> see [note] http://x.y") == "hello world see"
    assert prepare.clean_text("Café—déjà vu.") == "cafdj vu"
    assert prepare.clean_text("salt &amp; light") == "salt light"


def test_clean_lines_drops_empty():
    assert prepare.clean_lines("One line\n\n  \n123\nTwo line") == ["one line", "two line"]


@pytest.fixture
def tmp_project(tmp_path, monkeypatch):
    for name in ("DATA_DIR", "INTERIM_DIR", "MODELS_DIR", "RESULTS_DIR", "FIGURES_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    monkeypatch.setattr(config, "RAW_CSV", tmp_path / "raw.csv")
    monkeypatch.setattr(config, "POEMS_PKL", tmp_path / "poems.pkl")
    monkeypatch.setattr(config, "LINES_PKL", tmp_path / "lines.pkl")
    monkeypatch.setattr(config, "POEM_VECTORS", tmp_path / "vectors.npy")
    monkeypatch.setattr(config, "DOC2VEC_MODEL", tmp_path / "d2v.model")
    synthetic_csv(config.RAW_CSV)
    return tmp_path


def test_prepare_embed_cluster_topics(tmp_project):
    from poetry_nlp import cluster, embed, topics

    prepare.main(["--csv", str(config.RAW_CSV)])
    poems = pd.read_pickle(config.POEMS_PKL)
    lines = pd.read_pickle(config.LINES_PKL)
    assert len(poems) > 30 and len(lines) > len(poems)
    assert {"words", "lexical_diversity", "length_in_lines"} <= set(poems.columns)

    embed.main(["--vector-size", "16", "--epochs", "5", "--min-count", "1"])
    assert np.load(config.POEM_VECTORS).shape == (len(poems), 16)

    cluster.main(["--k-max", "6"])
    topics.main(["--topics", "3"])
    assert (config.RESULTS_DIR / "topics.json").exists()
    assert (config.FIGURES_DIR / "cluster_kmeans.png").exists()
