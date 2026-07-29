# %%
# CLIP vs DINOv2 RDM agreement, per site.
#
# 01/02 each correlate RDMs *within* one embedding space, across site pairs (e.g. Beijing
# CLIP vs San Jose CLIP). This script instead correlates *across* embedding spaces for the
# same site (e.g. Beijing CLIP vs Beijing DINOv2): do the two models agree on how the 12
# categories relate to each other, for the same drawings?
#
# Reuses the exact population/matched-subset machinery from 01/02: CLIP embeddings come
# straight from emb_df.parquet (../data/emb_df.parquet, produced by 01), DINOv2 embeddings
# are joined onto the same drawings by filename (as in 02).
#
# Outputs:
#   ../data/rdm_results_clip_vs_dinov2.csv          per-site CLIP-DINOv2 RDM correlations
#   ../data/figures/rdm_plots/stats/                per-site CLIP-vs-DINOv2 distance scatterplots
#
# Prerequisites:
#   ../data/emb_df.parquet                              (01_rdm_analysis_clip.py)
#   ../data/matched_subset_ids.csv                       (01_rdm_analysis_clip.py)
#   ../data/embeddings/full_embedding_store_dinov2.doc   (00_extract_dinov2_embeddings.py)

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")  # headless server -- save plots directly, no display
import matplotlib.pyplot as plt

from vislearnlabpy.embeddings.embedding_store import EmbeddingStore
from vislearnlabpy.embeddings.similarity_utils import cosine_matrix

CATEGORIES = [
    "airplane", "bike", "bird", "car", "cat", "chair",
    "cup", "hat", "house", "rabbit", "tree", "watch"
]
CATEGORIES_GROUPED = [
    "airplane", "bike", "car", "chair", "house", "tree",
    "cup", "hat", "watch",
    "bird", "cat", "rabbit",
]
SITE_NAMES = ['Beijing', 'San Jose', 'Kisumu', 'Delhi']

# %%
# CLIP population (with embeddings) from 01, DINOv2 embeddings joined by filename (as in 02)
clip_df = pd.read_parquet("../data/emb_df.parquet")
clip_df['location'] = clip_df['location'].replace({'New Delhi': 'Delhi'})
clip_df = clip_df.rename(columns={'embedding': 'embedding_clip'})
clip_df['filename'] = clip_df['url'].apply(lambda u: Path(u).name)

matched_ids = pd.read_csv("../data/matched_subset_ids.csv")
matched_ids['location'] = matched_ids['location'].replace({'New Delhi': 'Delhi'})
matched_urls = set(matched_ids['url'])

dinov2_store = EmbeddingStore.from_doc("../data/embeddings/full_embedding_store_dinov2")
dinov2_by_filename = {}
n_dupes = 0
for e in dinov2_store.EmbeddingList:
    fn = Path(e.url).name
    if fn in dinov2_by_filename:
        n_dupes += 1
    dinov2_by_filename[fn] = np.array(e.embedding)
if n_dupes:
    print(f"WARNING: {n_dupes} duplicate filenames in the DINOv2 store -- last one wins")

clip_df['embedding_dinov2'] = clip_df['filename'].map(dinov2_by_filename)
n_missing = clip_df['embedding_dinov2'].isna().sum()
if n_missing:
    print(f"WARNING: {n_missing} / {len(clip_df)} drawings have no DINOv2 embedding -- dropped")
df = clip_df.dropna(subset=['embedding_dinov2']).reset_index(drop=True)
print(f"Full population with both embeddings: {len(df)} / {len(clip_df)}")

matched_df = df[df['url'].isin(matched_urls)].reset_index(drop=True)
print(f"Matched subset with both embeddings: {len(matched_df)} / {len(matched_urls)}")

clip_dim = df['embedding_clip'].iloc[0].shape[0]
dinov2_dim = df['embedding_dinov2'].iloc[0].shape[0]

# %%
# RDM helpers -- same construction as 01/02 (pairwise cosine distance between mean category
# embeddings), parameterized by which embedding column to use

def compute_rdm(df, categories, emb_col, dim):
    means = np.stack([
        np.mean(np.stack(df[df['drawing_category'] == cat][emb_col].values), axis=0)
        if (df['drawing_category'] == cat).any() else np.zeros(dim)
        for cat in categories
    ])
    rdm = 1 - cosine_matrix(means, means)
    np.fill_diagonal(rdm, 0)
    return rdm

def rdm_from_embs_dict(embs_dict, categories, dim):
    means = np.stack([
        np.mean(embs_dict[cat], axis=0) if embs_dict.get(cat) else np.zeros(dim)
        for cat in categories
    ])
    rdm = 1 - cosine_matrix(means, means)
    np.fill_diagonal(rdm, 0)
    return rdm

def spearman_rdm(rdm1, rdm2):
    tril = np.tril_indices(len(rdm1), k=-1)
    r, _ = spearmanr(rdm1[tril], rdm2[tril])
    return r

def get_embs_by_category(df, categories, emb_col):
    return {cat: list(np.stack(df[df['drawing_category'] == cat][emb_col].values))
            for cat in categories if (df['drawing_category'] == cat).any()}

# %%
# observed CLIP-vs-DINOv2 RDM correlation per site -- full population and matched subset

def site_rdms_for(df, emb_col, dim):
    return {loc: compute_rdm(df[df['location'] == loc], CATEGORIES_GROUPED, emb_col, dim)
            for loc in SITE_NAMES}

clip_rdms_full = site_rdms_for(df, 'embedding_clip', clip_dim)
dinov2_rdms_full = site_rdms_for(df, 'embedding_dinov2', dinov2_dim)
clip_rdms_matched = site_rdms_for(matched_df, 'embedding_clip', clip_dim)
dinov2_rdms_matched = site_rdms_for(matched_df, 'embedding_dinov2', dinov2_dim)

obs_corrs_full = {loc: spearman_rdm(clip_rdms_full[loc], dinov2_rdms_full[loc]) for loc in SITE_NAMES}
obs_corrs_matched = {loc: spearman_rdm(clip_rdms_matched[loc], dinov2_rdms_matched[loc]) for loc in SITE_NAMES}

print("\nCLIP vs DINOv2 RDM correlation, per site:")
for loc in SITE_NAMES:
    print(f"{loc}: r_full = {obs_corrs_full[loc]:.3f}, r_matched = {obs_corrs_matched[loc]:.3f}")

# %%
# bootstrap 95% CIs (N=1000, paired resample within category x site cells -- the same
# resampled drawing indices are applied to both embedding spaces, since it's the same
# drawings represented two ways)

def bootstrap_clip_dino_corrs(df, site_names, categories, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    site_clip = {loc: get_embs_by_category(df[df['location'] == loc], categories, 'embedding_clip')
                 for loc in site_names}
    site_dino = {loc: get_embs_by_category(df[df['location'] == loc], categories, 'embedding_dinov2')
                 for loc in site_names}
    boot_corrs = {loc: [] for loc in site_names}
    tril = np.tril_indices(len(categories), k=-1)

    for _ in range(n_boot):
        for loc in site_names:
            boot_clip, boot_dino = {}, {}
            for cat in categories:
                clip_embs = site_clip[loc].get(cat, [])
                dino_embs = site_dino[loc].get(cat, [])
                n = len(clip_embs)
                idx = rng.integers(0, n, size=n) if n else np.array([], dtype=int)
                boot_clip[cat] = [clip_embs[i] for i in idx]
                boot_dino[cat] = [dino_embs[i] for i in idx]
            r, _ = spearmanr(
                rdm_from_embs_dict(boot_clip, categories, clip_dim)[tril],
                rdm_from_embs_dict(boot_dino, categories, dinov2_dim)[tril]
            )
            boot_corrs[loc].append(r)

    return {loc: np.array(vals) for loc, vals in boot_corrs.items()}

boot_results = bootstrap_clip_dino_corrs(df, SITE_NAMES, CATEGORIES_GROUPED)

print("\nBootstrap 95% CIs (full population):")
for loc, vals in boot_results.items():
    lo, hi = np.percentile(vals, [2.5, 97.5])
    print(f"{loc}: r = {obs_corrs_full[loc]:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]")

# %%
# category-label permutation test
# H0: CLIP-DINOv2 agreement isn't specific to category identity (sanity check, expect p ~ 0)

def category_perm_test(rdm1, rdm2, n_perm=1000, seed=42):
    rng = np.random.default_rng(seed)
    tril = np.tril_indices(len(rdm1), k=-1)
    obs_r, _ = spearmanr(rdm1[tril], rdm2[tril])
    perm_rs = np.array([
        spearmanr(rdm1[np.ix_(idx := rng.permutation(len(rdm1)), idx)][tril], rdm2[tril])[0]
        for _ in range(n_perm)
    ])
    return obs_r, perm_rs, float(np.mean(perm_rs >= obs_r))

print("\nCategory-label permutation (p = proportion of shuffled >= observed):")
category_perm_results = {}
for loc in SITE_NAMES:
    obs_r, _, p = category_perm_test(clip_rdms_full[loc], dinov2_rdms_full[loc])
    category_perm_results[loc] = (obs_r, p)
    print(f"{loc}: r = {obs_r:.3f}, p = {p:.4f}")

# %%
# summary CSV
summary_rows = []
for loc in SITE_NAMES:
    lo, hi = np.percentile(boot_results[loc], [2.5, 97.5])
    cat_r, cat_p = category_perm_results[loc]
    summary_rows.append({
        'site': loc,
        'r_full': obs_corrs_full[loc],
        'r_matched': obs_corrs_matched[loc],
        'boot_ci_lo': lo,
        'boot_ci_hi': hi,
        'category_perm_p': cat_p,
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("../data/rdm_results_clip_vs_dinov2.csv", index=False)
print("\nSaved summary to ../data/rdm_results_clip_vs_dinov2.csv")
print(summary_df.to_string(index=False))

# %%
# per-site scatterplots: CLIP distance vs DINOv2 distance for each category pair (full
# population), one point per off-diagonal cell -- visualizes the correlation above

rdm_plot_dir = Path("../data/figures/rdm_plots/stats")
rdm_plot_dir.mkdir(parents=True, exist_ok=True)
tril = np.tril_indices(len(CATEGORIES_GROUPED), k=-1)

for loc in SITE_NAMES:
    clip_d = clip_rdms_full[loc][tril]
    dino_d = dinov2_rdms_full[loc][tril]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(clip_d, dino_d, s=25, alpha=0.7, edgecolors='white', linewidth=0.5)
    ax.set(xlabel='CLIP cosine distance', ylabel='DINOv2 cosine distance',
           title=f'{loc}: CLIP vs DINOv2 (r = {obs_corrs_full[loc]:.3f})')
    plt.tight_layout()
    slug = loc.lower().replace(" ", "_")
    plt.savefig(rdm_plot_dir / f"rdm_clip_vs_dinov2_{slug}.svg", dpi=150, bbox_inches='tight')
    plt.close(fig)

print(f"\nSaved CLIP-vs-DINOv2 scatterplots to {rdm_plot_dir}/")
