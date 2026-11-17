"""Step 4: cluster the poem embeddings and score each method.

Follows the original analysis: standardise, L2-normalise and project the Doc2Vec
vectors onto two principal components, then compare K-means, DBSCAN,
agglomerative clustering and Gaussian mixtures on that plane. K-means is also
scored on the full normalised embedding for comparison.

Reads  data/interim/poem_vectors.npy
Writes data/interim/clusters.pkl, results/clustering.json, figures/cluster_*.png
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

from . import config
from .plotting import plt, savefig

SIL_SAMPLE = 5000


def sil(X, labels, seed=config.SEED, metric="euclidean"):
    if len(set(labels)) < 2:
        return float("nan")
    return float(silhouette_score(X, labels, sample_size=min(SIL_SAMPLE, len(X)),
                                  random_state=seed, metric=metric))


def project(vectors, seed=config.SEED):
    X_norm = normalize(StandardScaler().fit_transform(vectors))
    pca = PCA(n_components=2, random_state=seed)
    plane = pca.fit_transform(X_norm)
    pca99 = PCA(n_components=0.99, random_state=seed).fit(X_norm)
    return X_norm, plane, {
        "pc1_pc2_explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "components_for_99pct_variance": int(pca99.n_components_),
    }


def kmeans_sweep(plane, X_norm, k_range, seed=config.SEED):
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(plane)
        km_full = KMeans(n_clusters=k, n_init=2, random_state=seed).fit(X_norm)
        rows.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette_2d": sil(plane, km.labels_),
            "silhouette_full_cosine": sil(X_norm, km_full.labels_, metric="cosine"),
        })
    df = pd.DataFrame(rows)
    knee = KneeLocator(df["k"], df["inertia"], curve="convex", direction="decreasing").knee
    return df, int(knee)


def dbscan_grid(plane, eps_list, min_samples_list):
    rows = []
    for eps in eps_list:
        for ms in min_samples_list:
            labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(plane)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 2:
                continue
            rows.append({"eps": eps, "min_samples": ms, "n_clusters": n_clusters,
                         "noise": int((labels == -1).sum()), "silhouette": sil(plane, labels)})
    return pd.DataFrame(rows)


def knee_eps(plane):
    dist, _ = NearestNeighbors(n_neighbors=2).fit(plane).kneighbors(plane)
    d = np.sort(dist[:, 1])
    knee = KneeLocator(np.arange(len(d)), d, curve="convex", direction="increasing", S=3).knee
    return float(d[knee]), d


def gmm_bic(plane, seed=config.SEED):
    rows, best = [], None
    for cov in ("spherical", "tied", "diag", "full"):
        for n in range(1, 11):
            g = GaussianMixture(n_components=n, covariance_type=cov, random_state=seed).fit(plane)
            rows.append({"covariance": cov, "components": n, "bic": float(g.bic(plane))})
            if best is None or rows[-1]["bic"] < best[0]:
                best = (rows[-1]["bic"], g)
    return pd.DataFrame(rows), best[1]


def scatter(plane, labels, title, name, noise_label=None):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    labels = np.asarray(labels)
    if noise_label is not None:
        m = labels == noise_label
        ax.scatter(plane[m, 0], plane[m, 1], s=3, c="#bbbbbb", label="noise")
        plane, labels = plane[~m], labels[~m]
    sc = ax.scatter(plane[:, 0], plane[:, 1], s=3, c=labels, cmap="tab10")
    ax.legend(*sc.legend_elements(), title="cluster", loc="upper left", fontsize=8, markerscale=1.5)
    ax.set(title=title, xlabel="PC1", ylabel="PC2")
    savefig(fig, name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k-max", type=int, default=20)
    args = ap.parse_args(argv)
    config.ensure_dirs()
    warnings.filterwarnings("ignore", category=FutureWarning)

    vectors = np.load(config.POEM_VECTORS)
    X_norm, plane, pca_info = project(vectors)

    sweep, k_elbow = kmeans_sweep(plane, X_norm, range(2, args.k_max + 1))
    k_best_sil = int(sweep.loc[sweep["silhouette_2d"].idxmax(), "k"])
    km = KMeans(n_clusters=k_elbow, n_init=10, random_state=config.SEED).fit(plane)

    eps0, nn_dist = knee_eps(plane)
    eps_list = sorted({round(eps0, 3), 0.025, 0.05, 0.06, 0.09, 0.12, 0.15})
    db = dbscan_grid(plane, eps_list, [5, 10, 25, 50, 200])
    db_best = db.sort_values("silhouette", ascending=False).iloc[0] if len(db) else None
    db_labels = (DBSCAN(eps=db_best["eps"], min_samples=int(db_best["min_samples"])).fit_predict(plane)
                 if db_best is not None else np.zeros(len(plane), int))

    agg = AgglomerativeClustering(n_clusters=7, metric="cosine", linkage="complete").fit(plane)
    bic, gmm = gmm_bic(plane)
    gmm_labels = gmm.predict(plane)

    pd.DataFrame({"pc1": plane[:, 0], "pc2": plane[:, 1], "kmeans": km.labels_,
                  "dbscan": db_labels, "agglomerative": agg.labels_, "gmm": gmm_labels}
                 ).to_pickle(config.INTERIM_DIR / "clusters.pkl")

    result = {
        "projection": pca_info,
        "kmeans": {
            "elbow_k": k_elbow,
            "best_silhouette_k": k_best_sil,
            "silhouette_at_elbow_2d": round(float(sweep.set_index("k").loc[k_elbow, "silhouette_2d"]), 3),
            "sweep": sweep.round(4).to_dict(orient="records"),
        },
        "dbscan": {
            "knee_eps": round(eps0, 4),
            "best": None if db_best is None else {k: (v.item() if hasattr(v, "item") else v)
                                                  for k, v in db_best.round(4).items()},
            "grid": db.round(4).to_dict(orient="records"),
        },
        "agglomerative": {"n_clusters": 7, "linkage": "complete", "metric": "cosine",
                          "silhouette_2d": round(sil(plane, agg.labels_), 3)},
        "gmm": {"best_covariance": gmm.covariance_type, "best_components": int(gmm.n_components),
                "silhouette_2d": round(sil(plane, gmm_labels), 3),
                "bic": bic.round(1).to_dict(orient="records")},
    }
    (config.RESULTS_DIR / "clustering.json").write_text(json.dumps(result, indent=2))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(sweep["k"], sweep["inertia"], "o-", color="#3b6ea5")
    a1.axvline(k_elbow, ls="--", c="#c0392b", label=f"elbow k={k_elbow}")
    a1.set(title="K-means inertia", xlabel="k", ylabel="inertia")
    a1.legend()
    a2.plot(sweep["k"], sweep["silhouette_2d"], "o-", label="PCA plane")
    a2.plot(sweep["k"], sweep["silhouette_full_cosine"], "s-", label="full embedding (cosine)")
    a2.set(title="K-means silhouette", xlabel="k", ylabel="silhouette")
    a2.legend()
    savefig(fig, "cluster_kmeans_selection.png")

    scatter(plane, km.labels_, f"K-means, k={k_elbow} (elbow)", "cluster_kmeans.png")
    scatter(plane, db_labels, f"DBSCAN, eps={db_best['eps']}, min_samples={int(db_best['min_samples'])}"
            if db_best is not None else "DBSCAN", "cluster_dbscan.png", noise_label=-1)
    scatter(plane, gmm_labels, f"Gaussian mixture ({gmm.covariance_type}, {gmm.n_components} components)",
            "cluster_gmm.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (cov, grp) in enumerate(bic.groupby("covariance", sort=False)):
        ax.bar(grp["components"] + 0.2 * (i - 1.5), grp["bic"], width=0.2, label=cov)
    ax.set(title="Gaussian mixture BIC (lower is better)", xlabel="components", ylabel="BIC")
    ax.legend()
    savefig(fig, "cluster_gmm_bic.png")

    print(json.dumps({k: v for k, v in result["kmeans"].items() if k != "sweep"}, indent=2))
    print("dbscan best:", result["dbscan"]["best"])
    print("agglomerative:", result["agglomerative"]["silhouette_2d"],
          "gmm:", result["gmm"]["best_covariance"], result["gmm"]["best_components"])


if __name__ == "__main__":
    main()
