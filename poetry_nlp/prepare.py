"""Step 1: clean the Kaggle CSV into poem-level and line-level tables.

Reads  data/raw/kaggle_poem_dataset.csv
Writes data/interim/poems.pkl, data/interim/lines.pkl,
       results/corpus_stats.json, figures/corpus_*.png
"""
import argparse
import html
import json
import re
import string

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .plotting import plt, savefig

_SPACES = re.compile(r" +")
_BRACKETS = re.compile(r"\[.*?\]")
_URLS = re.compile(r"https?://\S+|www\.\S+")
_TAGS = re.compile(r"<.*?>+")
_DIGIT_WORDS = re.compile(r"\w*\d\w*")
_SPLIT = re.compile(r"\s|[!,.?]")


def clean_text(text: str) -> str:
    """Lower-case one line and strip markup, URLs, digits, punctuation and non-ASCII."""
    text = html.unescape(text).lower()
    text = _SPACES.sub(" ", text)
    text = text.replace("%", "")
    text = _BRACKETS.sub("", text)
    text = _URLS.sub("", text)
    text = _TAGS.sub("", text)
    text = _DIGIT_WORDS.sub("", text)
    tokens = _SPLIT.split(text)
    text = " ".join(t.strip(string.punctuation) for t in tokens)
    text = text.encode("ascii", "ignore").decode()
    return _SPACES.sub(" ", text).strip()


def clean_lines(content: str) -> list:
    """Split a poem into cleaned, non-empty lines."""
    lines = (clean_text(line) for line in content.split("\n") if line)
    return [line for line in lines if line]


def lexical_diversity(words) -> float:
    return len(set(words)) / len(words) if len(words) else 0.0


def load_raw(path=config.RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df = df.drop(columns=[c for c in ("unnamed: 0", "poetry foundation id") if c in df.columns])
    df["author"] = df["author"].astype("category")
    return df


def build_poems(raw: pd.DataFrame, stats_out: dict) -> pd.DataFrame:
    stats_out["raw_poems"] = int(len(raw))
    df = (raw.drop_duplicates(subset=["content"])
             .drop_duplicates(subset=["author", "title"])
             .dropna()
             .copy())
    stats_out["deduplicated_poems"] = int(len(df))
    stats_out["authors"] = int(df["author"].nunique())

    df["line"] = df["content"].map(clean_lines)
    df["length_in_lines"] = df["line"].map(len)

    # Drop length outliers (|z| >= 1.5) and anything shorter than three lines.
    df = df[np.abs(stats.zscore(df["length_in_lines"])) < 1.5]
    df = df[df["length_in_lines"] > 2].copy()

    df["words"] = df["line"].map(lambda lines: " ".join(lines).split())
    df["lexical_diversity"] = df["words"].map(lexical_diversity)
    df["word_lengths"] = df["words"].map(lambda ws: [len(w) for w in ws])
    df["max_word_length"] = df["word_lengths"].map(lambda ls: max(ls) if ls else 0)

    # Very long "words" are almost always run-together tokens from scraping.
    df = df[(df["max_word_length"] < 20) & (df["max_word_length"] > 0)].copy()
    stats_out["clean_poems"] = int(len(df))
    return df.drop(columns=["content"])


def build_lines(poems: pd.DataFrame, stats_out: dict) -> pd.DataFrame:
    lines = (poems[["author", "title", "line"]]
             .explode("line")
             .dropna(subset=["line"])
             .reset_index(drop=True))
    lines["line_no"] = lines.groupby(["author", "title"], observed=True).cumcount() + 1
    lines["words"] = lines["line"].str.split()
    lines["length_in_words"] = lines["words"].map(len)
    stats_out["raw_lines"] = int(len(lines))

    lines = lines[np.abs(stats.zscore(lines["length_in_words"])) < 2].copy()
    lines["lexical_diversity"] = lines["words"].map(lexical_diversity)
    lines["max_word_length"] = lines["words"].map(lambda ws: max(len(w) for w in ws))
    lines = lines[stats.zscore(lines["max_word_length"]) < 2].copy()
    stats_out["clean_lines"] = int(len(lines))
    return lines.reset_index(drop=True)


def describe(df: pd.DataFrame, cols) -> dict:
    d = df[cols].describe().round(3)
    return {c: d[c].to_dict() for c in cols}


def plot_corpus(poems: pd.DataFrame, lines: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(poems["length_in_lines"], bins=60, color="#3b6ea5")
    axes[0].set(title="Poem length", xlabel="lines per poem", ylabel="poems")
    axes[1].hist(lines["length_in_words"], bins=range(1, lines["length_in_words"].max() + 2),
                 color="#3b6ea5")
    axes[1].set(title="Line length", xlabel="words per line", ylabel="lines")
    axes[2].hist(poems["lexical_diversity"], bins=60, color="#3b6ea5")
    axes[2].set(title="Lexical diversity", xlabel="unique words / words (per poem)", ylabel="poems")
    savefig(fig, "corpus_distributions.png")

    by_author = lines.groupby("author", observed=True)[["lexical_diversity", "length_in_words"]].mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hexbin(by_author["lexical_diversity"], by_author["length_in_words"], gridsize=40,
              cmap="Blues", mincnt=1)
    ax.set(title="Authors: average line length vs. line-level diversity",
           xlabel="mean lexical diversity of a line", ylabel="mean words per line")
    savefig(fig, "corpus_authors.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(config.RAW_CSV))
    args = ap.parse_args(argv)
    config.ensure_dirs()

    summary = {}
    raw = load_raw(args.csv)
    poems = build_poems(raw, summary)
    lines = build_lines(poems, summary)
    summary["poem_stats"] = describe(poems, ["length_in_lines", "lexical_diversity", "max_word_length"])
    summary["line_stats"] = describe(lines, ["length_in_words", "lexical_diversity", "line_no"])

    poems.to_pickle(config.POEMS_PKL)
    lines.to_pickle(config.LINES_PKL)
    (config.RESULTS_DIR / "corpus_stats.json").write_text(json.dumps(summary, indent=2))
    plot_corpus(poems, lines)
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("_stats")}, indent=2))


if __name__ == "__main__":
    main()
