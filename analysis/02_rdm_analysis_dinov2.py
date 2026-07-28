# %%
# Re-run the CLIP RDM/permutation/downsample pipeline (rdm_analysis_clip.py) with DINOv2
# embeddings substituted for the exact same drawing population and matched subset.
#
# The matched-subset selection (which 2,392 drawings, at which ages/categories/sites) is
# NOT re-derived here -- DINOv2 has no text tower, so there's no DINOv2-native
# target_similarity to re-run match_recognizability() against, and re-matching on DINOv2
# data would select a different subset, defeating the point of holding the population fixed.
# Instead this script reuses the exact drawing identities rdm_analysis_clip.py persists
# (../data/emb_df.parquet for the full population, ../data/matched_subset_ids.csv for the
# matched subset) and only substitutes each drawing's embedding vector with its DINOv2
# counterpart, joined by filename. Any difference in RDM correlations, permutation
# p-values, or downsample behavior below is therefore attributable to the embedding space
# alone, not to a different set of drawings.
#
# Prerequisites:
#   ../data/emb_df.parquet               (rdm_analysis_clip.py)
#   ../data/matched_subset_ids.csv        (rdm_analysis_clip.py, RDMs with matching...)
#   ../data/embeddings/full_embedding_store_dinov2.doc  (extract_dinov2_embeddings.py)

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
# full CLIP-derived population -- this defines the population and the exact age/category/
# site splits; only the embedding vector gets substituted below, everything else about
# which drawings exist and how they're labeled comes straight from the CLIP run
clip_meta = pd.read_parquet("../data/emb_df.parquet").drop(columns=['embedding'])
matched_ids = pd.read_csv("../data/matched_subset_ids.csv")

# tolerate stale persisted data from before rdm_analysis_clip.py renamed this site
# "New Delhi" -> "Delhi"; a no-op once emb_df.parquet has been regenerated
clip_meta['location'] = clip_meta['location'].replace({'New Delhi': 'Delhi'})
matched_ids['location'] = matched_ids['location'].replace({'New Delhi': 'Delhi'})
matched_urls = set(matched_ids['url'])

print(f"Full population (from emb_df.parquet): {len(clip_meta)} drawings")
print(f"Matched subset (from matched_subset_ids.csv): {len(matched_ids)} drawings")
print(f"Matched subset age range: {matched_ids['age'].min()}-{matched_ids['age'].max()}")
print("Matched subset per site x category cell counts (CLIP, for reference):")
print(matched_ids.groupby(['location', 'drawing_category']).size().unstack().to_string())

# %%
# DINOv2 embeddings, joined by filename rather than full url -- the CLIP urls were
# extracted with a /Volumes mount prefix and the DINOv2 urls with a /labs mount prefix, so
# only the basename is guaranteed to agree between the two runs
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

dinov2_dim = next(iter(dinov2_by_filename.values())).shape[0]
print(f"DINOv2 store: {len(dinov2_by_filename)} unique filenames, dim={dinov2_dim}")

# %%
clip_meta = clip_meta.copy()
clip_meta['filename'] = clip_meta['url'].apply(lambda u: Path(u).name)
clip_meta['embedding'] = clip_meta['filename'].map(dinov2_by_filename)

n_missing = clip_meta['embedding'].isna().sum()
if n_missing:
    print(f"WARNING: {n_missing} / {len(clip_meta)} drawings have no DINOv2 embedding -- dropped")
emb_df = clip_meta.dropna(subset=['embedding']).reset_index(drop=True)
print(f"Full population with DINOv2 embeddings: {len(emb_df)} / {len(clip_meta)}")

matched_df = emb_df[emb_df['url'].isin(matched_urls)].reset_index(drop=True)
n_matched_missing = len(matched_urls) - len(matched_df)
if n_matched_missing:
    print(f"WARNING: {n_matched_missing} / {len(matched_urls)} matched-subset drawings "
          f"have no DINOv2 embedding -- dropped, matched subset is no longer exactly identical")
print(f"Matched subset with DINOv2 embeddings: {len(matched_df)} / {len(matched_urls)}")
print("Matched subset per site x category cell counts (DINOv2):")
print(matched_df.groupby(['location', 'drawing_category']).size().unstack().to_string())

# %%
# RDM / permutation / downsample helpers -- identical to rdm_analysis_clip.py
# (compute_rdm, category_perm_test, site_perm_test, random_downsample), except the
# embedding-dim zero-fallback is inferred from the data instead of hardcoded to CLIP's 512

def compute_rdm(df, categories):
    means = np.stack([
        np.mean(np.stack(df[df['drawing_category'] == cat]['embedding'].values), axis=0)
        if (df['drawing_category'] == cat).any() else np.zeros(dinov2_dim)
        for cat in categories
    ])
    rdm = 1 - cosine_matrix(means, means)
    np.fill_diagonal(rdm, 0)
    return rdm

def rdm_from_embs_dict(embs_dict, categories):
    means = np.stack([
        np.mean(embs_dict[cat], axis=0) if embs_dict.get(cat) else np.zeros(dinov2_dim)
        for cat in categories
    ])
    rdm = 1 - cosine_matrix(means, means)
    np.fill_diagonal(rdm, 0)
    return rdm

def spearman_rdm(rdm1, rdm2):
    tril = np.tril_indices(len(rdm1), k=-1)
    r, _ = spearmanr(rdm1[tril], rdm2[tril])
    return r

def get_embs_by_category(df, categories):
    return {cat: list(np.stack(df[df['drawing_category'] == cat]['embedding'].values))
            for cat in categories if (df['drawing_category'] == cat).any()}

def category_perm_test(rdm1, rdm2, n_perm=1000, seed=42):
    """1a -- shuffle category labels on one RDM. H0: correlation isn't specific to category identity."""
    rng = np.random.default_rng(seed)
    tril = np.tril_indices(len(rdm1), k=-1)
    obs_r, _ = spearmanr(rdm1[tril], rdm2[tril])
    perm_rs = np.array([
        spearmanr(rdm1[np.ix_(idx := rng.permutation(len(rdm1)), idx)][tril], rdm2[tril])[0]
        for _ in range(n_perm)
    ])
    return obs_r, perm_rs, float(np.mean(perm_rs >= obs_r))

def site_perm_test(loc1, loc2, emb_df, categories, n_perm=1000, seed=42):
    """1b -- pool two sites' embeddings and re-split randomly. H0: the site boundary is meaningless."""
    rng = np.random.default_rng(seed)
    tril = np.tril_indices(len(categories), k=-1)
    embs1 = get_embs_by_category(emb_df[emb_df['location'] == loc1], categories)
    embs2 = get_embs_by_category(emb_df[emb_df['location'] == loc2], categories)
    obs_r, _ = spearmanr(
        rdm_from_embs_dict(embs1, categories)[tril],
        rdm_from_embs_dict(embs2, categories)[tril]
    )
    perm_rs = []
    for _ in range(n_perm):
        pe1, pe2 = {}, {}
        for cat in categories:
            all_e = embs1.get(cat, []) + embs2.get(cat, [])
            n1 = len(embs1.get(cat, []))
            if all_e:
                idx = rng.permutation(len(all_e))
                pe1[cat] = [all_e[i] for i in idx[:n1]]
                pe2[cat] = [all_e[i] for i in idx[n1:]]
            else:
                pe1[cat] = pe2[cat] = []
        r, _ = spearmanr(rdm_from_embs_dict(pe1, categories)[tril],
                         rdm_from_embs_dict(pe2, categories)[tril])
        perm_rs.append(r)
    perm_rs = np.array(perm_rs)
    return obs_r, perm_rs, float(np.mean(perm_rs <= obs_r))

def random_downsample(emb_df, target_cell_sizes, site_names, categories, n_iter=100, seed=42):
    """control: random downsampling to the matched subset's cell sizes, no similarity matching."""
    rng = np.random.default_rng(seed)
    all_corrs = {pair: [] for pair in combinations(site_names, 2)}
    for _ in range(n_iter):
        sampled_indices = []
        for loc in site_names:
            for cat in categories:
                cell = emb_df[(emb_df['location'] == loc) & (emb_df['drawing_category'] == cat)]
                n_target = target_cell_sizes.get((loc, cat), 0)
                if n_target == 0 or len(cell) == 0:
                    continue
                n_target = min(n_target, len(cell))
                sampled = rng.choice(cell.index, size=n_target, replace=False)
                sampled_indices.extend(sampled.tolist())
        sampled_df = emb_df.loc[sampled_indices]
        sampled_rdms = {
            loc: compute_rdm(sampled_df[sampled_df['location'] == loc], CATEGORIES_GROUPED)
            for loc in site_names
        }
        for s1, s2 in combinations(site_names, 2):
            all_corrs[(s1, s2)].append(spearman_rdm(sampled_rdms[s1], sampled_rdms[s2]))
    return {pair: np.array(vals) for pair, vals in all_corrs.items()}

# %%
# RDM construction + observed Spearman correlations -- full population (DINOv2)
site_rdms = {loc: compute_rdm(emb_df[emb_df['location'] == loc], CATEGORIES_GROUPED) for loc in SITE_NAMES}
obs_corrs = {}
print("\nFull-population RDM correlations (DINOv2):")
for s1, s2 in combinations(SITE_NAMES, 2):
    r = spearman_rdm(site_rdms[s1], site_rdms[s2])
    obs_corrs[(s1, s2)] = r
    print(f"{s1} vs {s2}: r = {r:.3f}")

# %%
# 1a -- category-label permutation test (on full-population RDMs, mirrors cell 26)
print("\n1a - category-label permutation (p = proportion of shuffled >= observed):")
category_perm_results = {}
for (s1, s2) in obs_corrs:
    obs_r, _, p = category_perm_test(site_rdms[s1], site_rdms[s2])
    category_perm_results[(s1, s2)] = (obs_r, p)
    print(f"{s1} vs {s2}: r = {obs_r:.3f}, p = {p:.4f}")

# %%
# 1b -- site-label permutation test (on full population, mirrors cell 27)
print("\n1b - site-label permutation (p = proportion of null <= observed):")
site_perm_results = {}
for s1, s2 in combinations(SITE_NAMES, 2):
    obs_r, perm_rs, p = site_perm_test(s1, s2, emb_df, CATEGORIES)
    site_perm_results[(s1, s2)] = (obs_r, perm_rs.mean(), p)
    print(f"{s1} vs {s2}: r = {obs_r:.3f}, null mean = {perm_rs.mean():.3f}, p = {p:.4f}")

# %%
# matched-subset RDMs -- DINOv2 embeddings substituted into the exact CLIP-matched drawings
matched_rdms = {
    loc: compute_rdm(matched_df[matched_df['location'] == loc], CATEGORIES_GROUPED)
    for loc in SITE_NAMES
}
print("\nRDM correlations, matched subset (DINOv2), vs full population (DINOv2):")
matched_corrs = {}
for s1, s2 in combinations(SITE_NAMES, 2):
    r_matched = spearman_rdm(matched_rdms[s1], matched_rdms[s2])
    matched_corrs[(s1, s2)] = r_matched
    print(f"{s1} vs {s2}: r = {r_matched:.3f} (full = {obs_corrs[(s1, s2)]:.3f})")

# %%
# 1b (matched subset) -- site-label permutation test restricted to the matched subset,
# same H0 as the full-population version (cell 27): the site boundary is meaningless --
# any random partition of the same sizes should give equally correlated RDMs
print("\n1b (matched subset) - site-label permutation (p = proportion of null <= observed):")
site_perm_matched_results = {}
for s1, s2 in combinations(SITE_NAMES, 2):
    obs_r, perm_rs, p = site_perm_test(s1, s2, matched_df, CATEGORIES)
    site_perm_matched_results[(s1, s2)] = (obs_r, perm_rs.mean(), p)
    print(f"{s1} vs {s2}: r = {obs_r:.3f}, null mean = {perm_rs.mean():.3f}, p = {p:.4f}")

# %%
# random-downsample control -- re-run rather than assumed to carry over from the CLIP run,
# since the null distribution can behave differently in a space with different intrinsic
# dimensionality (DINOv2's 768 vs CLIP's 512)
matched_cell_sizes = matched_df.groupby(['location', 'drawing_category']).size()
random_corrs = random_downsample(emb_df, matched_cell_sizes, SITE_NAMES, CATEGORIES)

print("\nRandom downsample control (DINOv2, mean +/- sd over 100 iterations):")
downsample_results = {}
for s1, s2 in combinations(SITE_NAMES, 2):
    rand_mean = random_corrs[(s1, s2)].mean()
    rand_sd = random_corrs[(s1, s2)].std()
    downsample_results[(s1, s2)] = (rand_mean, rand_sd)
    print(f"{s1} vs {s2}: full = {obs_corrs[(s1,s2)]:.3f}, matched = {matched_corrs[(s1,s2)]:.3f}, "
          f"random = {rand_mean:.3f} (sd {rand_sd:.3f})")

# %%
# tidy summary for direct comparison against the CLIP run's printed output in
# rdm_analysis_clip.py
summary_rows = []
for s1, s2 in combinations(SITE_NAMES, 2):
    cat_r, cat_p = category_perm_results[(s1, s2)]
    site_r, site_null_mean, site_p = site_perm_results[(s1, s2)]
    site_r_matched, site_null_mean_matched, site_p_matched = site_perm_matched_results[(s1, s2)]
    rand_mean, rand_sd = downsample_results[(s1, s2)]
    summary_rows.append({
        'site1': s1, 'site2': s2,
        'r_full_dinov2': obs_corrs[(s1, s2)],
        'r_matched_dinov2': matched_corrs[(s1, s2)],
        'r_random_downsample_mean_dinov2': rand_mean,
        'r_random_downsample_sd_dinov2': rand_sd,
        'category_perm_p_dinov2': cat_p,
        'site_perm_p_dinov2': site_p,
        'site_perm_null_mean_dinov2': site_null_mean,
        'site_perm_p_matched_dinov2': site_p_matched,
        'site_perm_null_mean_matched_dinov2': site_null_mean_matched,
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("../data/rdm_results_dinov2.csv", index=False)
print("\nSaved summary to ../data/rdm_results_dinov2.csv")
print(summary_df.to_string(index=False))

# %%
# RDM heatmaps -- full population and matched subset (DINOv2), mirrors the RDM heatmap
# section of rdm_analysis_clip.py, shared color scale across all of them
rdm_plot_dir = Path("../data/figures/rdm_plots/stats")
rdm_plot_dir.mkdir(parents=True, exist_ok=True)

all_rdms = list(site_rdms.values()) + list(matched_rdms.values())
vmax = max(rdm[np.tril_indices(len(CATEGORIES_GROUPED), k=-1)].max() for rdm in all_rdms)

def plot_rdm(rdm, title, filename):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(rdm, cmap='viridis', vmin=0, vmax=vmax)
    ax.set_xticks(range(len(CATEGORIES_GROUPED)))
    ax.set_yticks(range(len(CATEGORIES_GROUPED)))
    ax.set_xticklabels(CATEGORIES_GROUPED, rotation=90, fontsize=8)
    ax.set_yticklabels(CATEGORIES_GROUPED, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label='Cosine distance')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

for loc in SITE_NAMES:
    slug = loc.lower().replace(" ", "_")
    plot_rdm(site_rdms[loc], f'{loc} (DINOv2, full)', rdm_plot_dir / f"rdm_dinov2_{slug}.svg")
    plot_rdm(matched_rdms[loc], f'{loc} (DINOv2, matched)', rdm_plot_dir / f"rdm_matched_dinov2_{slug}.svg")

print(f"\nSaved RDM heatmaps to {rdm_plot_dir}/")
