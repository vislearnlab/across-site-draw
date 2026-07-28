# %%
# UMAP of rabbit drawings (CLIP embeddings), rendered with each drawing's own
# ink recolored by site instead of a plain dot. Complementary to
# umap_rabbit_by_location.py (dots) and to the across-site-draw-explorer
# (https://github.com/vislearnlab/across-site-draw-explorer), which renders
# drawings from raw stroke path data fetched from a remote server -- we only
# have the rasterized PNG thumbnails locally, so ink is recolored on the
# pixels directly: dark ink -> alpha, background -> transparent, then tinted
# to the site's hex color.
#
# Prerequisite: ../data/emb_df.parquet (produced by rdm_analysis_clip.py).
# Drawing PNGs are read from each row's `url`, which must be reachable on
# disk (e.g. the /Volumes/vislearnlab mount).

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Patch
from PIL import Image
import umap

# %%
emb_df = pd.read_parquet("../data/emb_df.parquet")
rabbit_df = emb_df[emb_df['drawing_category'] == 'rabbit'].reset_index(drop=True)
print(f"Rabbit drawings: {len(rabbit_df)}")
print(rabbit_df['location'].value_counts())

# %%
embeddings = np.stack(rabbit_df['embedding'].values)

reducer = umap.UMAP(n_components=2, random_state=42)
coords = reducer.fit_transform(embeddings)
rabbit_df['umap_x'] = coords[:, 0]
rabbit_df['umap_y'] = coords[:, 1]

# %%
location_colors = {
    'Beijing': '#1b9e77',
    'San Jose': '#e7298a',
    'Kisumu': '#7570b3',
    'Delhi': '#d95f02',
}

def resolve_drawing_path(url):
    """`url` was recorded at CLIP-extraction time as an absolute /Volumes/vislearnlab/...
    path; on machines where the same share is mounted at /labs/vislearnlab/... instead
    (e.g. the DINOv2 extraction host, see extract_dinov2_embeddings.py), fall back to
    that prefix. Returns None if neither mount has the file."""
    p = Path(url)
    if p.exists():
        return p
    alt = Path(url.replace("/Volumes/", "/labs/", 1))
    if alt.exists():
        return alt
    return None

def tint_drawing(path, hex_color):
    """Recolor a drawing's ink to hex_color; ink darkness becomes alpha, background
    becomes transparent. Works for both binary tablet strokes (Beijing/San
    Jose/Delhi) and grayscale Kisumu scans."""
    rgb = tuple(int(hex_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
    gray = np.array(Image.open(path).convert('L'), dtype=np.float32)
    alpha = np.clip(255 - gray, 0, 255).astype(np.uint8)
    rgba = np.zeros((*gray.shape, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = rgb
    rgba[..., 3] = alpha
    return Image.fromarray(rgba, mode='RGBA')

# %%
fig, ax = plt.subplots(figsize=(11, 10))

n_missing = 0
for _, row in rabbit_df.iterrows():
    path = resolve_drawing_path(row['url'])
    if path is None:
        n_missing += 1
        continue
    tinted = tint_drawing(path, location_colors[row['location']])
    imagebox = OffsetImage(tinted, zoom=0.25)
    ab = AnnotationBbox(imagebox, (row['umap_x'], row['umap_y']), frameon=False, pad=0)
    ax.add_artist(ab)

if n_missing:
    print(f"WARNING: {n_missing} drawing(s) missing on disk at their recorded url, skipped")

pad_x = (rabbit_df['umap_x'].max() - rabbit_df['umap_x'].min()) * 0.05
pad_y = (rabbit_df['umap_y'].max() - rabbit_df['umap_y'].min()) * 0.05
ax.set_xlim(rabbit_df['umap_x'].min() - pad_x, rabbit_df['umap_x'].max() + pad_x)
ax.set_ylim(rabbit_df['umap_y'].min() - pad_y, rabbit_df['umap_y'].max() + pad_y)
ax.set(xlabel='UMAP 1', ylabel='UMAP 2',
       title=f'UMAP of rabbit drawings (n={len(rabbit_df)})')
ax.set_xticks([])
ax.set_yticks([])

legend_elements = [Patch(facecolor=c, label=loc) for loc, c in location_colors.items()]
ax.legend(handles=legend_elements, title='Location', loc='best')

plt.tight_layout()

output_dir = Path("../data/figures/umap_plots")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "umap_rabbit_strokes_by_location.png", dpi=600, bbox_inches='tight')
# plt.savefig(output_dir / "umap_rabbit_strokes_by_location.svg", bbox_inches='tight')
plt.show()

# %%
# same UMAP space (rabbit_df/coords above, fit only once) and the same
# xlim/ylim/figsize/dpi/zoom as the plot above, but saved as three separate
# plots split by age group instead of one plot with everyone -- so the three
# images are on the same visual scale and line up consistently in a row
age_groups = [(4, 5), (6, 7), (8, 9)]

for age_lo, age_hi in age_groups:
    group_df = rabbit_df[rabbit_df['age'].between(age_lo, age_hi)]

    fig, ax = plt.subplots(figsize=(11, 10))

    n_missing = 0
    for _, row in group_df.iterrows():
        path = resolve_drawing_path(row['url'])
        if path is None:
            n_missing += 1
            continue
        tinted = tint_drawing(path, location_colors[row['location']])
        imagebox = OffsetImage(tinted, zoom=0.25)
        ab = AnnotationBbox(imagebox, (row['umap_x'], row['umap_y']), frameon=False, pad=0)
        ax.add_artist(ab)

    if n_missing:
        print(f"WARNING: age {age_lo}-{age_hi}: {n_missing} drawing(s) missing on disk, skipped")

    ax.set_xlim(rabbit_df['umap_x'].min() - pad_x, rabbit_df['umap_x'].max() + pad_x)
    ax.set_ylim(rabbit_df['umap_y'].min() - pad_y, rabbit_df['umap_y'].max() + pad_y)
    ax.set(xlabel='UMAP 1', ylabel='UMAP 2',
           title=f'UMAP of rabbit drawings, age {age_lo}-{age_hi} (n={len(group_df)})')
    ax.set_xticks([])
    ax.set_yticks([])

    legend_elements = [Patch(facecolor=c, label=loc) for loc, c in location_colors.items()]
    ax.legend(handles=legend_elements, title='Location', loc='best')

    plt.tight_layout()
    plt.savefig(output_dir / f"umap_rabbit_strokes_age_{age_lo}_{age_hi}.png", dpi=600, bbox_inches='tight')
    plt.show()
