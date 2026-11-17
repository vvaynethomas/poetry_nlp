"""Step 3: train a Doc2Vec (distributed bag-of-words) model and embed every poem.

Reads  data/interim/poems.pkl
Writes models/poem_doc2vec_dbow.model, data/interim/poem_vectors.npy,
       results/embedding.json
"""
import argparse
import json
import multiprocessing

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from . import config


def train(word_lists, vector_size=500, epochs=50, min_count=2, seed=config.SEED, workers=None):
    docs = [TaggedDocument(words, [i]) for i, words in enumerate(word_lists)]
    model = Doc2Vec(vector_size=vector_size, dm=0, min_count=min_count, epochs=epochs,
                    alpha=0.025, min_alpha=0.0001, seed=seed,
                    workers=workers or max(1, multiprocessing.cpu_count() - 1))
    model.build_vocab(docs)
    model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def self_retrieval(model, word_lists, sample=500, seed=config.SEED):
    """Share of poems whose re-inferred vector has the poem itself as nearest neighbour."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(word_lists), size=min(sample, len(word_lists)), replace=False)
    top1 = top10 = 0
    for i in idx:
        sims = model.dv.most_similar([model.infer_vector(word_lists[i])], topn=10)
        ids = [tag for tag, _ in sims]
        top1 += ids[0] == i
        top10 += i in ids
    return top1 / len(idx), top10 / len(idx)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vector-size", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--min-count", type=int, default=2)
    args = ap.parse_args(argv)
    config.ensure_dirs()

    poems = pd.read_pickle(config.POEMS_PKL)
    word_lists = poems["words"].tolist()
    model = train(word_lists, args.vector_size, args.epochs, args.min_count)
    model.save(str(config.DOC2VEC_MODEL))

    vectors = np.vstack([model.dv[i] for i in range(len(word_lists))])
    np.save(config.POEM_VECTORS, vectors)

    top1, top10 = self_retrieval(model, word_lists)
    result = {
        "poems": len(word_lists),
        "vector_size": args.vector_size,
        "epochs": args.epochs,
        "vocabulary": len(model.wv.key_to_index),
        "self_retrieval_top1": round(top1, 3),
        "self_retrieval_top10": round(top10, 3),
    }
    (config.RESULTS_DIR / "embedding.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
