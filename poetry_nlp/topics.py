"""Step 5: describe what the clusters contain and fit an NMF topic model.

For every K-means cluster: size, length, lexical diversity, part-of-speech mix and
the terms that are most over-represented relative to the whole corpus. Separately,
a 12-topic NMF model over TF-IDF gives a vocabulary-level view of the corpus.

Reads  data/interim/poems.pkl, data/interim/clusters.pkl, data/interim/poem_pos.pkl (optional)
Writes results/topics.json, results/cluster_profiles.csv, figures/topics_*.png
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from . import config
from .plotting import plt, savefig

# Archaic and poetic function words that the standard list misses.
EXTRA_STOP = {"thee", "thou", "thy", "thine", "ye", "o", "oh", "shall", "hath", "doth", "tis",
              "like", "just", "don", "didn", "let", "said", "say", "came", "come", "went", "got",
              "ll", "ve", "re", "isn", "wasn", "couldn", "wouldn", "ain", "upon", "yes", "know"}
STOP = sorted(ENGLISH_STOP_WORDS | EXTRA_STOP)


def tfidf(texts, min_df=20, max_df=0.5):
    vec = TfidfVectorizer(stop_words=STOP, min_df=min_df, max_df=max_df,
                          token_pattern=r"(?u)\b[a-z][a-z]{2,}\b", sublinear_tf=True)
    return vec, vec.fit_transform(texts)


def distinctive_terms(X, terms, labels, top=10):
    overall = np.asarray(X.mean(axis=0)).ravel()
    out = {}
    for c in sorted(set(labels)):
        m = np.asarray(X[labels == c].mean(axis=0)).ravel()
        lift = (m + 1e-4) / (overall + 1e-4)
        score = lift * m  # favour terms that are both frequent and over-represented
        out[int(c)] = [terms[i] for i in np.argsort(score)[::-1][:top]]
    return out


def nmf_topics(X, terms, n_topics=12, top=8, seed=config.SEED):
    nmf = NMF(n_components=n_topics, init="nndsvda", random_state=seed, max_iter=400)
    W = nmf.fit_transform(X)
    topics = [[terms[i] for i in comp.argsort()[::-1][:top]] for comp in nmf.components_]
    return W, topics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topics", type=int, default=12)
    args = ap.parse_args(argv)
    config.ensure_dirs()

    poems = pd.read_pickle(config.POEMS_PKL).reset_index(drop=True)
    clusters = pd.read_pickle(config.INTERIM_DIR / "clusters.pkl")
    labels = clusters["kmeans"].to_numpy()
    texts = poems["words"].map(" ".join)

    large = len(texts) >= 1000  # the thresholds below only make sense for the full corpus
    vec, X = tfidf(texts, min_df=20 if large else 1, max_df=0.5 if large else 1.0)
    terms = vec.get_feature_names_out()

    prof = pd.DataFrame({
        "cluster": labels,
        "length_in_lines": poems["length_in_lines"],
        "lexical_diversity": poems["lexical_diversity"],
        "words_per_line": poems["words"].map(len) / poems["length_in_lines"],
        "pc1": clusters["pc1"],
    })
    pos_path = config.INTERIM_DIR / "poem_pos.pkl"
    if pos_path.exists():
        pos = pd.read_pickle(pos_path).reset_index(drop=True)
        for col in ("NOUN", "VERB", "ADJ", "PRON"):
            prof[col.lower() + "_share"] = pos[col]
    profiles = prof.groupby("cluster").mean().round(3)
    profiles.insert(0, "poems", prof.groupby("cluster").size())
    profiles["distinctive_terms"] = pd.Series(distinctive_terms(X, terms, labels)).map(", ".join)
    profiles = profiles.sort_values("pc1")
    profiles.to_csv(config.RESULTS_DIR / "cluster_profiles.csv")

    W, topics = nmf_topics(X, terms, args.topics)
    dominant = W.argmax(axis=1)
    topic_share = np.bincount(dominant, minlength=args.topics) / len(dominant)
    crosstab = pd.crosstab(labels, dominant, normalize="index").round(3)

    result = {
        "tfidf_vocabulary": int(len(terms)),
        "nmf_topics": [{"topic": i, "share_of_poems": round(float(topic_share[i]), 3), "terms": t}
                       for i, t in enumerate(topics)],
        "cluster_profiles": json.loads(profiles.reset_index().to_json(orient="records")),
        "cluster_topic_mix": {int(c): {int(t): float(v) for t, v in row.items() if v >= 0.1}
                              for c, row in crosstab.iterrows()},
    }
    (config.RESULTS_DIR / "topics.json").write_text(json.dumps(result, indent=2))

    fig, ax = plt.subplots(figsize=(9, 0.45 * args.topics + 1))
    order = np.argsort(topic_share)
    ax.barh([", ".join(topics[i][:5]) for i in order], topic_share[order], color="#3b6ea5")
    ax.set(title="NMF topics: share of poems where each topic dominates", xlabel="share of poems")
    savefig(fig, "topics_nmf.png")

    numeric = profiles.drop(columns=["distinctive_terms", "poems", "pc1"])
    z = (numeric - prof[numeric.columns].mean()) / prof[numeric.columns].std()
    fig, ax = plt.subplots(figsize=(1.1 * len(numeric.columns) + 3, 0.5 * len(z) + 1.5))
    im = ax.imshow(z.to_numpy(), cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(z.columns)), z.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(z)), [f"cluster {c} (n={n})" for c, n in profiles["poems"].items()])
    fig.colorbar(im, ax=ax, label="difference from corpus mean (SD)")
    ax.set_title("K-means clusters, ordered along PC1")
    savefig(fig, "topics_cluster_profiles.png")

    print(profiles.to_string())
    for t in result["nmf_topics"]:
        print(t["topic"], t["share_of_poems"], " ".join(t["terms"]))


if __name__ == "__main__":
    main()
