import json, os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.metrics.pairwise import cosine_similarity

import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# ------------------------------- load artifacts ------------------------------
@st.cache_data(show_spinner=False)
def load_artifacts(artifacts_dir: str):
    adir = Path(artifacts_dir)
    df = pd.read_parquet(adir / "papers.parquet")
    X = np.load(adir / "paper_embeddings.npy")
    with (adir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    edges = pd.read_csv(adir / "edges.csv")
    return df, X, meta, edges

# ------------------------------ pyvis renderer -------------------------------
@st.cache_data(show_spinner=False)
def render_pyvis(df: pd.DataFrame, edges: pd.DataFrame, height: str = "700px"):
    net = Network(height=height, width="100%", bgcolor="#0b1221", font_color="#e2e8f0")
    net.barnes_hut()

    # add nodes
    for idx, row in df.iterrows():
        label = row['paper_id']
        title = ("<b>" + row['paper_id'] + "</b><br>" + ", ".join(row['keywords'][:6])).replace("\n"," ")
        color = px.colors.qualitative.Plotly[row['cluster'] % 10]
        net.add_node(int(idx), label=label, title=title, color=color)

    # add edges (sparsify lightly for speed)
    for _, r in edges.iterrows():
        if r['weight'] < 0.35:  # drop very weak links
            continue
        net.add_edge(int(r['src']), int(r['dst']), value=float(r['weight']))

    html = net.generate_html(notebook=False)
    return html

# ----------------------------------- UI --------------------------------------
st.set_page_config(page_title="SpaceBio Knowledge Engine", layout="wide")
st.title("🛰️ SpaceBio Knowledge Engine")

with st.sidebar:
    st.header("Artifacts")
    artifacts_dir = st.text_input("Path to artifacts directory", value="./artifacts")
    if st.button("Load", type="primary"):
        st.session_state['loaded'] = True

if 'loaded' not in st.session_state:
    st.info("Enter your artifacts path in the sidebar and click Load.")
    st.stop()

# Load
try:
    df, X, meta, edges = load_artifacts(artifacts_dir)
except Exception as e:
    st.error(f"Failed to load artifacts: {e}")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header("Explore")
    show = st.radio("View", ["Constellations (2D)", "Graph (Network)", "Search"], index=0)
    cluster_sel = st.multiselect("Filter clusters", sorted(df['cluster'].unique()), default=[])
    topn = st.slider("Top-N neighbors to highlight (2D)", 0, 20, 0)

# Filters
mask = df['cluster'].isin(cluster_sel) if cluster_sel else np.ones(len(df), dtype=bool)
view_df = df.loc[mask].copy()

# ===== View: Constellations (UMAP/TSNE) ======================================
if show == "Constellations (2D)":
    st.subheader("Constellations of Papers")
    fig = px.scatter(
        view_df,
        x="x", y="y",
        color="cluster",
        hover_data={"paper_id": True, "cluster": True, "x": False, "y": False},
        height=700,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Optional neighbor highlight on selection
    sel_pid = st.selectbox("Inspect a paper", view_df['paper_id'].tolist())
    if sel_pid:
        i = int(df.index[df['paper_id'] == sel_pid][0])
        sims = cosine_similarity(X[i:i+1], X).ravel()
        nn_idx = sims.argsort()[::-1][1: topn+1] if topn > 0 else []
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"**Nearest neighbors to `{sel_pid}`**")
            for j in nn_idx:
                st.write(f"{df.loc[j, 'paper_id']} — sim={sims[j]:.3f}")
        with cols[1]:
            st.markdown("**Keywords**")
            st.write(", ".join(df.loc[i, 'keywords'][:12]))

# ===== View: Graph (interactive network) =====================================
elif show == "Graph (Network)":
    st.subheader("Similarity Network (KNN)")
    html = render_pyvis(df if view_df.shape[0]==df.shape[0] else view_df.reset_index(drop=True),
                        edges)
    components.html(html, height=720, scrolling=True)

# ===== View: Search ===========================================================
else:
    st.subheader("Semantic Search")
    q = st.text_input("Enter keywords (will match against TF‑IDF keywords & titles)")
    k = st.slider("Top-K", 5, 50, 10)

    # Simple hybrid: keyword match on paper_id/keywords + centroid similarity to keyword centroid (approx)
    if st.button("Search", type="primary") and q.strip():
        q_terms = set([t.strip().lower() for t in q.split() if t.strip()])
        # boost scores when query terms appear in keywords or id
        base = np.zeros(len(df), dtype=float)
        for i, row in df.iterrows():
            text = (row['paper_id'] + " " + " ".join(row['keywords'])).lower()
            overlap = len(q_terms.intersection(text.split()))
            base[i] = overlap
        # normalize and combine with mean similarity to papers that contain terms
        if base.max() > 0:
            base = base / (base.max() + 1e-9)
        # fallback: purely keyword rank
        order = np.argsort(-base)[:k]
        res = df.iloc[order][['paper_id','cluster','keywords']]
        st.dataframe(res, use_container_width=True)

    st.markdown("---")
    st.markdown("### Paper Details")
    pid = st.selectbox("Select a paper to view details", df['paper_id'].tolist())
    if pid:
        row = df.loc[df['paper_id'] == pid].iloc[0]
        st.markdown(f"**Paper ID:** `{pid}`  ")
        st.markdown(f"**Cluster:** {row['cluster']}  ")
        st.markdown("**Top keywords:** " + ", ".join(row['keywords']))
        with st.expander("Raw text (first ~1500 chars)"):
            st.write((row['text'] or "")[:1500] + ("…" if len(row['text'])>1500 else ""))
