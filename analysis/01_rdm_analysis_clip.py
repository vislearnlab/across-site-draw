# %%
# CLIP RDM analysis (201b, Analysis 1) -- drawing variability across sites.
#
# Builds the full drawing population from stored CLIP embedding docs, computes CLIP
# recognizability, and produces two sets of category RDMs per site: the full population
# and a recognizability-matched subset (target_similarity distributions matched across
# sites within each category).
#
# Outputs:
#   ../data/emb_df.parquet               full population incl. embeddings + recognizability
#   ../data/matched_subset_ids.csv        recognizability-matched subset drawing identities
#   ../data/figures/rdm_plots/stats/      RDM heatmaps, full population and matched subset
#
# RDM correlations, permutation tests, and downsample control results are printed to
# stdout. rerun_rdm_dinov2.py substitutes DINOv2 embeddings for this exact population
# (emb_df.parquet + matched_subset_ids.csv) and reruns the same analyses.

from pathlib import Path
import re, os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless server -- save plots directly, no display
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from vislearnlabpy.embeddings.embedding_store import EmbeddingStore
from vislearnlabpy.embeddings.similarity_utils import cosine_matrix

# %%
embedding_dir = Path("../data/embeddings")

beijing_store = EmbeddingStore.from_doc(str(embedding_dir / "beijing_store"))
kisumu_store  = EmbeddingStore.from_doc(str(embedding_dir / "kisumu_store"))
sanjose_store = EmbeddingStore.from_doc(str(embedding_dir / "sanjose_store"))
delhi_store   = EmbeddingStore.from_doc(str(embedding_dir / "newdelhi_store"))

# %%
text_store = EmbeddingStore.from_doc(str(embedding_dir / "text_embeddings"))
text_embs = {}
for emb in text_store.EmbeddingList:
    label = emb.text.lower().strip()
    text_embs[label] = np.array(emb.embedding)

print(list(text_embs.keys()))

# %%
# Build drawing-level dataframe
#
# Each embedding store has `.text` (drawing category) and `.url` (original file path). Age
# and participant_id are parsed from the URL. Delhi age is looked up from the existing
# recognizability CSV.

CATEGORIES = [
    "airplane", "bike", "bird", "car", "cat", "chair",
    "cup", "hat", "house", "rabbit", "tree", "watch"
]

CATEGORIES_GROUPED = [
    # big objects
    "airplane", "bike", "car", "chair", "house", "tree",
    # small objects
    "cup", "hat", "watch",
    # animals
    "bird", "cat", "rabbit",
]

# metadata parsers
# URL format differs by site, yikes

def parse_beijing_url(url):
    fn = os.path.basename(url)
    age = int(re.search(r'age(\d+)', fn).group(1))
    location = "Beijing" if "THU" in fn else "USA"
    participant_id = fn.removesuffix('.png').split('_')[-1]
    return {'location': location, 'age': age, 'participant_id': participant_id}

def parse_kisumu_url(url):
    fn = os.path.basename(url)
    # age in filename is offset by 3 years
    age = int(re.search(r'age(\d+)', fn).group(1)) + 3
    participant_id = fn.split('_')[0]
    return {'location': 'Kisumu', 'age': age, 'participant_id': participant_id}

def parse_india_url(url, age_lookup):
    fn = os.path.basename(url)
    for prefix in ('a_', 'an_', 'three_', 'two_'):
        if fn.startswith(prefix):
            fn = fn[len(prefix):]
            break
    parts = fn.split('_')
    participant_id = parts[2].upper()
    return {'location': 'India', 'age': age_lookup.get(participant_id, np.nan), 'participant_id': participant_id}

# Delhi age lookup from existing CSV
_meta = pd.read_csv("../data/clip_recognizability.csv")
india_age_lookup = (
    _meta[_meta['location'] == 'India']
    .drop_duplicates('participant_id')
    .set_index('participant_id')['age']
    .to_dict()
)

# per-site dataframes from stores
def build_store_df(store, parse_fn, **kwargs):
    rows = []
    for emb in store.EmbeddingList:
        if emb.text not in CATEGORIES:
            continue
        meta = parse_fn(emb.url, **kwargs)
        # url (original file path/basename) is kept as the per-drawing identifier so any
        # downstream subset (e.g. matched_df) can be re-joined against a different embedding
        # space (e.g. DINOv2) by filename rather than relying on (location, age,
        # participant_id, drawing_category), which is not guaranteed unique
        rows.append({'embedding': np.array(emb.embedding), 'url': emb.url, 'drawing_category': emb.text, **meta})
    return pd.DataFrame(rows)

emb_df = pd.concat([
    build_store_df(beijing_store, parse_beijing_url),
    build_store_df(sanjose_store, parse_beijing_url),   # same URL format as Beijing
    build_store_df(kisumu_store, parse_kisumu_url),
    build_store_df(delhi_store, parse_india_url, age_lookup=india_age_lookup),
], ignore_index=True)

emb_df = emb_df[(emb_df['age'] >= 4) & (emb_df['age'] <= 9)].reset_index(drop=True)
emb_df['location'] = emb_df['location'].replace({'USA': 'San Jose', 'India': 'Delhi'})
print(emb_df.groupby('location').size())
print(f"Total: {len(emb_df)} drawings")

# %%
# CLIP recognizability -- target_similarity drives the recognizability-matched subset below

text_labels = list(text_embs.keys())
text_matrix = np.stack([text_embs[l] for l in text_labels])

CLIP_LOGIT_SCALE = 100.0  # matches CLIP's learned logit scale (exp(temperature) ~= 100)

def classify_drawing(drawing_emb, target_category):
    sims = cosine_matrix(drawing_emb.reshape(1, -1), text_matrix)[0]
    predicted = text_labels[np.argmax(sims)]
    target_key = f"drawing of a {target_category}"
    target_idx = text_labels.index(target_key)
    logits = CLIP_LOGIT_SCALE * sims
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    recognized = predicted == target_key
    return recognized, float(probs[target_idx])

results = emb_df.apply(
    lambda row: classify_drawing(row['embedding'], row['drawing_category']), axis=1
)
emb_df['recognized'] = results.apply(lambda x: x[0])
emb_df['target_similarity'] = results.apply(lambda x: x[1])

# %%
# persist the full emb_df (incl. embedding + url) for reuse by other notebooks/scripts
# (e.g. word_embeddings/compute_alignment.ipynb, and joining a different embedding space's
# stores against the exact same drawing identities via url)
emb_df.to_parquet("../data/emb_df.parquet")

# %%
# RDM analysis (Analysis 1)
#
# 12x12 category RDMs per site (pairwise cosine distances between mean category
# embeddings). Spearman-correlated across all 6 site pairs, with bootstrap 95% CIs and two
# permutation tests.

def compute_rdm(df, categories):
    means = np.stack([
        np.mean(np.stack(df[df['drawing_category'] == cat]['embedding'].values), axis=0)
        if (df['drawing_category'] == cat).any() else np.zeros(512)
        for cat in categories
    ])
    rdm = 1 - cosine_matrix(means, means)
    np.fill_diagonal(rdm, 0)
    return rdm

def rdm_from_embs_dict(embs_dict, categories):
    means = np.stack([
        np.mean(embs_dict[cat], axis=0) if embs_dict.get(cat) else np.zeros(512)
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

# %%
SITE_NAMES = ['Beijing', 'San Jose', 'Kisumu', 'Delhi']

# %%
# compute RDMs + observed correlations -- full population
site_rdms = {loc: compute_rdm(emb_df[emb_df['location'] == loc], CATEGORIES_GROUPED) for loc in SITE_NAMES}

obs_corrs = {}
print("\nFull-population RDM correlations:")
for s1, s2 in combinations(SITE_NAMES, 2):
    r = spearman_rdm(site_rdms[s1], site_rdms[s2])
    obs_corrs[(s1, s2)] = r
    print(f"{s1} vs {s2}: r = {r:.3f}")

# %%
# bootstrap 95% CIs (N=1000, resample within category x site cells)

def bootstrap_rdm_corrs(emb_df, site_names, categories, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    site_embs = {loc: get_embs_by_category(emb_df[emb_df['location'] == loc], categories)
                 for loc in site_names}
    pairs = list(combinations(site_names, 2))
    boot_corrs = {pair: [] for pair in pairs}
    tril = np.tril_indices(len(categories), k=-1)

    for _ in range(n_boot):
        boot_rdms = {}
        for loc in site_names:
            boot_embs = {}
            for cat in categories:
                embs = site_embs[loc].get(cat, [])
                n = len(embs)
                boot_embs[cat] = [embs[i] for i in rng.integers(0, n, size=n)] if n else []
            boot_rdms[loc] = rdm_from_embs_dict(boot_embs, categories)
        for (s1, s2) in pairs:
            r, _ = spearmanr(boot_rdms[s1][tril], boot_rdms[s2][tril])
            boot_corrs[(s1, s2)].append(r)

    return {pair: np.array(vals) for pair, vals in boot_corrs.items()}

boot_results = bootstrap_rdm_corrs(emb_df, SITE_NAMES, CATEGORIES)

print("\nBootstrap 95% CIs:")
for (s1, s2), vals in boot_results.items():
    lo, hi = np.percentile(vals, [2.5, 97.5])
    print(f"{s1} vs {s2}: r = {obs_corrs[(s1,s2)]:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]")

# %%
# 1a -- category-label permutation test
# H0: RDM correlation is not specific to category identity (sanity check, expect p ~ 0)

def category_perm_test(rdm1, rdm2, n_perm=1000, seed=42):
    rng = np.random.default_rng(seed)
    tril = np.tril_indices(len(rdm1), k=-1)
    obs_r, _ = spearmanr(rdm1[tril], rdm2[tril])
    perm_rs = np.array([
        spearmanr(rdm1[np.ix_(idx := rng.permutation(len(rdm1)), idx)][tril], rdm2[tril])[0]
        for _ in range(n_perm)
    ])
    return obs_r, perm_rs, float(np.mean(perm_rs >= obs_r))

print("\n1a - category-label permutation (p = proportion of shuffled >= observed):")
category_perm_results = {}
for (s1, s2) in obs_corrs:
    obs_r, _, p = category_perm_test(site_rdms[s1], site_rdms[s2])
    category_perm_results[(s1, s2)] = (obs_r, p)
    print(f"{s1} vs {s2}: r = {obs_r:.3f}, p = {p:.4f}")

# %%
# 1b -- site-label permutation test
# H0: site boundary is meaningless -- any random partition of same sizes gives equally
# correlated RDMs

def site_perm_test(loc1, loc2, emb_df, categories, n_perm=1000, seed=42):
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

print("\n1b - site-label permutation (p = proportion of null <= observed):")
site_perm_results = {}
for s1, s2 in combinations(SITE_NAMES, 2):
    obs_r, perm_rs, p = site_perm_test(s1, s2, emb_df, CATEGORIES)
    site_perm_results[(s1, s2)] = (obs_r, perm_rs.mean(), p)
    print(f"{s1} vs {s2}: r = {obs_r:.3f}, null mean = {perm_rs.mean():.3f}, p = {p:.4f}")

# %%
# RDMs with matching target_similarity distributions across sites

def match_recognizability(emb_df, site_names, categories, n_bins=5, seed=42):
    """within each category, match target_similarity distributions across sites."""
    rng = np.random.default_rng(seed)
    matched_indices = []

    for cat in categories:
        cat_df = emb_df[emb_df['drawing_category'] == cat]
        bin_edges = np.quantile(cat_df['target_similarity'], np.linspace(0, 1, n_bins + 1))
        bin_edges[0] -= 1e-6
        cat_df = cat_df.copy()
        cat_df['sim_bin'] = pd.cut(cat_df['target_similarity'], bins=bin_edges, labels=False)

        for b in range(n_bins):
            bin_indices = {}
            for loc in site_names:
                idx = cat_df[(cat_df['location'] == loc) & (cat_df['sim_bin'] == b)].index
                bin_indices[loc] = idx

            min_count = min(len(idx) for idx in bin_indices.values())
            if min_count == 0:
                continue

            for loc in site_names:
                if len(bin_indices[loc]) > min_count:
                    kept = rng.choice(bin_indices[loc], size=min_count, replace=False)
                    matched_indices.extend(kept.tolist())
                else:
                    matched_indices.extend(bin_indices[loc].tolist())

    return emb_df.loc[matched_indices].copy()

matched_df = match_recognizability(emb_df, SITE_NAMES, CATEGORIES)

print("\nAfter matching:")
print(f"Total drawings: {len(matched_df)} (was {len(emb_df)})")
print("\nRecognizability by site (matched):")
print(matched_df.groupby('location')['recognized'].mean())
print("\nMean target similarity by site (matched):")
print(matched_df.groupby('location')['target_similarity'].mean())
print("\nDrawings per site-category cell:")
print(matched_df.groupby(['location', 'drawing_category']).size().unstack().to_string())

matched_rdms = {
    loc: compute_rdm(matched_df[matched_df['location'] == loc], CATEGORIES_GROUPED)
    for loc in SITE_NAMES
}

print("\nRDM correlations after recognizability matching:")
for s1, s2 in combinations(SITE_NAMES, 2):
    r_matched = spearman_rdm(matched_rdms[s1], matched_rdms[s2])
    r_orig = obs_corrs[(s1, s2)]
    print(f"{s1} vs {s2}: r = {r_matched:.3f} (was {r_orig:.3f})")

# persist the exact matched-subset drawing identities (url, site, age, category, and the
# CLIP recognizability values that drove the matching) so this recognizability-matched
# subset can be reproduced/reused without rerunning match_recognizability -- e.g. to
# substitute a different embedding space's vectors for these same drawings and re-run the
# RDM/permutation/downsample pipeline on an identical population. match_recognizability is
# deterministic (seed=42) given the same emb_df, so this file and a fresh rerun should agree.
matched_df.drop(columns=['embedding']).to_csv("../data/matched_subset_ids.csv", index=False)
print(f"\nPersisted {len(matched_df)} matched-subset drawing IDs to ../data/matched_subset_ids.csv")

# %%
# 1b (matched subset) -- site-label permutation test restricted to the matched subset,
# same H0 as the full-population version above: the site boundary is meaningless -- any
# random partition of the same sizes should give equally correlated RDMs

print("\n1b (matched subset) - site-label permutation (p = proportion of null <= observed):")
site_perm_matched_results = {}
for s1, s2 in combinations(SITE_NAMES, 2):
    obs_r, perm_rs, p = site_perm_test(s1, s2, matched_df, CATEGORIES)
    site_perm_matched_results[(s1, s2)] = (obs_r, perm_rs.mean(), p)
    print(f"{s1} vs {s2}: r = {obs_r:.3f}, null mean = {perm_rs.mean():.3f}, p = {p:.4f}")

# %%
# control: random downsampling to same cell sizes, no similarity matching

matched_cell_sizes = matched_df.groupby(['location', 'drawing_category']).size()

def random_downsample(emb_df, target_cell_sizes, site_names, categories, n_iter=100, seed=42):
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

random_corrs = random_downsample(emb_df, matched_cell_sizes, SITE_NAMES, CATEGORIES)

print("\nRandom downsample control (mean +/- sd over 100 iterations):")
downsample_results = {}
for s1, s2 in combinations(SITE_NAMES, 2):
    rand_mean = random_corrs[(s1, s2)].mean()
    rand_sd = random_corrs[(s1, s2)].std()
    downsample_results[(s1, s2)] = (rand_mean, rand_sd)
    r_matched = spearman_rdm(matched_rdms[s1], matched_rdms[s2])
    r_orig = obs_corrs[(s1, s2)]
    print(f"{s1} vs {s2}: original = {r_orig:.3f}, matched = {r_matched:.3f}, "
          f"random = {rand_mean:.3f} (sd {rand_sd:.3f})")

# %%
# tidy summary for direct comparison against the DINOv2 run's printed output in
# 02_rdm_analysis_dinov2.py
summary_rows = []
for s1, s2 in combinations(SITE_NAMES, 2):
    cat_r, cat_p = category_perm_results[(s1, s2)]
    site_r, site_null_mean, site_p = site_perm_results[(s1, s2)]
    site_r_matched, site_null_mean_matched, site_p_matched = site_perm_matched_results[(s1, s2)]
    rand_mean, rand_sd = downsample_results[(s1, s2)]
    r_matched = spearman_rdm(matched_rdms[s1], matched_rdms[s2])
    summary_rows.append({
        'site1': s1, 'site2': s2,
        'r_full_clip': obs_corrs[(s1, s2)],
        'r_matched_clip': r_matched,
        'r_random_downsample_mean_clip': rand_mean,
        'r_random_downsample_sd_clip': rand_sd,
        'category_perm_p_clip': cat_p,
        'site_perm_p_clip': site_p,
        'site_perm_null_mean_clip': site_null_mean,
        'site_perm_p_matched_clip': site_p_matched,
        'site_perm_null_mean_matched_clip': site_null_mean_matched,
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("../data/rdm_results_clip.csv", index=False)
print("\nSaved summary to ../data/rdm_results_clip.csv")
print(summary_df.to_string(index=False))

# %%
# RDM heatmaps -- full population and matched subset, shared color scale across all of them

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
    plot_rdm(site_rdms[loc], f'{loc} (full)', rdm_plot_dir / f"rdm_{slug}.svg")
    plot_rdm(matched_rdms[loc], f'{loc} (matched)', rdm_plot_dir / f"rdm_matched_{slug}.svg")

print(f"\nSaved RDM heatmaps to {rdm_plot_dir}/")