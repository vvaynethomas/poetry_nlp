"""Step 2: part-of-speech profile of every poem with spaCy's tagger.

Reads  data/interim/poems.pkl
Writes data/interim/poem_pos.pkl (per-poem share of each coarse POS),
       results/pos_stats.json, figures/pos_*.png
"""
import argparse
import json
from collections import Counter

import pandas as pd

from . import config
from .plotting import plt, savefig

COARSE = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP", "DET", "CCONJ", "PROPN", "AUX"]


def load_tagger():
    import spacy
    return spacy.load("en_core_web_sm",
                      exclude=["parser", "lemmatizer", "ner"])


def tag_poems(texts, n_process=1, batch_size=64):
    nlp = load_tagger()
    nlp.max_length = max(nlp.max_length, max((len(t) for t in texts), default=0) + 1)
    fine, coarse = [], []
    for doc in nlp.pipe(texts, n_process=n_process, batch_size=batch_size):
        fine.append(Counter(t.tag_ for t in doc))
        coarse.append(Counter(t.pos_ for t in doc))
    return fine, coarse


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-process", type=int, default=2)
    args = ap.parse_args(argv)
    config.ensure_dirs()

    poems = pd.read_pickle(config.POEMS_PKL)
    texts = poems["line"].map(" ".join).tolist()
    fine, coarse = tag_poems(texts, n_process=args.n_process)

    shares = pd.DataFrame(coarse, index=poems.index).fillna(0)
    shares = shares.div(shares.sum(axis=1), axis=0)
    shares = shares.reindex(columns=COARSE, fill_value=0.0)
    shares.to_pickle(config.INTERIM_DIR / "poem_pos.pkl")

    total_fine = sum(fine, Counter())
    total_coarse = sum(coarse, Counter())
    n = sum(total_coarse.values())
    result = {
        "tokens_tagged": int(n),
        "coarse_share": {k: round(v / n, 4) for k, v in total_coarse.most_common()},
        "top_fine_tags": dict(total_fine.most_common(20)),
        "median_poem_share": shares.median().round(4).to_dict(),
    }
    (config.RESULTS_DIR / "pos_stats.json").write_text(json.dumps(result, indent=2))

    tags, counts = zip(*total_fine.most_common(20))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(tags, counts, color="#3b6ea5")
    ax.set(title="Most frequent Penn Treebank tags across the corpus", ylabel="tokens")
    ax.tick_params(axis="x", rotation=60)
    savefig(fig, "pos_top_tags.png")
    print(json.dumps(result["coarse_share"], indent=2))


if __name__ == "__main__":
    main()
