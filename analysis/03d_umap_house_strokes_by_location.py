# %%
# UMAP of house drawings (CLIP embeddings), rendered with each drawing's own
# ink recolored by site instead of a plain dot. Same treatment as
# 03b_umap_rabbit_strokes_by_location.py, applied to the house category.
#
# Prerequisite: ../data/emb_df.parquet (produced by rdm_analysis_clip.py).
# Drawing PNGs are read from each row's `url`, which must be reachable on
# disk (e.g. the /Volumes/vislearnlab mount, or /labs/vislearnlab on the
# DINOv2 extraction host).

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
house_df = emb_df[emb_df['drawing_category'] == 'house'].reset_index(drop=True)
print(f"House drawings: {len(house_df)}")
print(house_df['location'].value_counts())

# %%
embeddings = np.stack(house_df['embedding'].values)

reducer = umap.UMAP(n_components=2, random_state=42)
coords = reducer.fit_transform(embeddings)
house_df['umap_x'] = coords[:, 0]
house_df['umap_y'] = coords[:, 1]

# %%
location_colors = {
    'Beijing': '#66c2a5',
    'San Jose': '#e78ac3',
    'Kisumu': '#8da0cb',
    'Delhi': '#fc8d62',
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
for _, row in house_df.iterrows():
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

pad_x = (house_df['umap_x'].max() - house_df['umap_x'].min()) * 0.05
pad_y = (house_df['umap_y'].max() - house_df['umap_y'].min()) * 0.05
ax.set_xlim(house_df['umap_x'].min() - pad_x, house_df['umap_x'].max() + pad_x)
ax.set_ylim(house_df['umap_y'].min() - pad_y, house_df['umap_y'].max() + pad_y)
ax.set(xlabel='UMAP 1', ylabel='UMAP 2',
       title=f'UMAP of house drawings (n={len(house_df)})')
ax.set_xticks([])
ax.set_yticks([])

legend_elements = [Patch(facecolor=c, label=loc) for loc, c in location_colors.items()]
ax.legend(handles=legend_elements, title='Location', loc='best')

plt.tight_layout()

output_dir = Path("../data/figures/umap_plots")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "umap_house_strokes_by_location.png", dpi=600, bbox_inches='tight')
# plt.savefig(output_dir / "umap_house_strokes_by_location.svg", bbox_inches='tight')
plt.show()