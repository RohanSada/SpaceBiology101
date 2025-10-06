import argparse, json, os, math, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

import networkx as nx
import community as community_louvain  # python-louvain


try:
    import umap
except Exception:
    umap = None

# ------------------------ text normalization helpers -------------------------
_token_re = re.compile(r"[A-Za-z][A-Za-z\-']+")

def normalize_text(s: str) -> str:
    s = s or ""
    s = s.lower()
    return " ".join(_token_re.findall(s))

# --------------------------- aggregation routines ----------------------------

def load_chunks(jsonl_path: Path):
    texts_by_paper = defaultdict(list)
    embeds_by_paper = defaultdict(list)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading JSONL"):
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = obj["paper_id"]
            texts_by_paper[pid].append(obj.get("text", ""))
            embeds_by_paper[pid].append(obj.get("embedding", []))

    # convert to arrays / strings
    paper_ids = []
    paper_texts = []
    paper_embeds = []

    for pid in texts_by_paper:
        chunks_text = "\n".join(texts_by_paper[pid])
        vecs = np.array(embeds_by_paper[pid], dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[0] == 0:
            continue
        centroid = vecs.mean(axis=0)
        # L2 normalize for cosine space
        norm = np.linalg.norm(centroid) + 1e-12
        centroid = centroid / norm
        paper_ids.append(pid)
        paper_texts.append(chunks_text)
        paper_embeds.append(centroid)

    X = np.vstack(paper_embeds)
    df = pd.DataFrame({"paper_id": paper_ids, "text": paper_texts})
    return df, X

# ---------------------------- keyword extraction -----------------------------

def extract_keywords(texts, top_k=12, max_features=50000):
    # Light‑weight, no LLM. Works across 608 papers.
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1,2),
        min_df=2,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-']+\b",
    )
    Xtfidf = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())

    top_terms = []
    for i in range(Xtfidf.shape[0]):
        row = Xtfidf.getrow(i)
        if row.nnz == 0:
            top_terms.append([])
            continue
        idx = np.argsort(row.data)[::-1][:top_k]
        top_terms.append(terms[row.indices[idx]].tolist())
    return top_terms

# ---------------------------- graph construction -----------------------------

def build_graph(embeds: np.ndarray, k: int = 12):
    # KNN in cosine space (embeds assumed L2 normalized)
    nbrs = NearestNeighbors(n_neighbors=min(k+1, embeds.shape[0]), metric="cosine")
    nbrs.fit(embeds)
    distances, indices = nbrs.kneighbors(embeds)

    G = nx.Graph()
    n = embeds.shape[0]
    for i in range(n):
        G.add_node(i)

    # skip self (indices[:,0] is self)
    for i in range(n):
        for j, d in zip(indices[i,1:], distances[i,1:]):
            w = 1.0 - float(d)  # cosine similarity
            if w <= 0:
                continue
            # undirected; keep max weight if duplicate
            if G.has_edge(i, j):
                G[i][j]['weight'] = max(G[i][j]['weight'], w)
            else:
                G.add_edge(i, j, weight=w)
    return G

# ------------------------------ dimensionality -------------------------------

def embed_2d(embeds: np.ndarray, seed: int = 42):
    if umap is None:
        from sklearn.manifold import TSNE
        proj = TSNE(n_components=2, init='random', random_state=seed, perplexity=30)
        Y = proj.fit_transform(embeds)
    else:
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.05, metric='cosine', random_state=seed)
        Y = reducer.fit_transform(embeds)
    return Y

# ---------------------------------- main -------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--k", type=int, default=12, help="K for KNN graph")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df, X = load_chunks(Path(args.jsonl))

    # keywords (no LLM)
    kw = extract_keywords([normalize_text(t) for t in df["text"].tolist()], top_k=12)
    df["keywords"] = kw

    # graph
    G = build_graph(X, k=args.k)

    # Louvain clustering
    partition = community_louvain.best_partition(G, weight='weight', resolution=1.0, random_state=42)
    df['cluster'] = [partition[i] for i in range(len(df))]

    # 2D projection for "constellations"
    Y = embed_2d(X)
    df['x'] = Y[:,0]
    df['y'] = Y[:,1]

    # Save artifacts
    df.to_parquet(outdir / "papers.parquet", index=False)

    # edges csv for pyvis
    rows = []
    for u, v, data in G.edges(data=True):
        rows.append({"src": int(u), "dst": int(v), "weight": float(data.get('weight', 0.0))})
    pd.DataFrame(rows).to_csv(outdir / "edges.csv", index=False)

    # also save numpy embeddings for search
    np.save(outdir / "paper_embeddings.npy", X)

    # small metadata json (index mapping)
    meta = {"index_to_paper_id": df["paper_id"].tolist()}
    with (outdir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved: {outdir}/papers.parquet, edges.csv, paper_embeddings.npy, meta.json")