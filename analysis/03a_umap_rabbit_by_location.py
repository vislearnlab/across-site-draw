# %%
# UMAP of rabbit drawings (CLIP embeddings) within the rabbit-only subspace,
# colored by location. Complementary to the cross-category t-SNE views in
# embedding_retrieval.ipynb -- this fits UMAP on a single category's embeddings
# only, rather than projecting all categories together, so structure is not
# dominated by between-category separation.
#
# Prerequisite: ../data/emb_df.parquet (produced by rdm_analysis_clip.py).

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# fixed color order (matches the site-coloring convention used elsewhere in
# this repo, e.g. embedding_retrieval.ipynb's create_tsne_per_category)
location_colors = {
    'Beijing': '#2a78d6',
    'San Jose': '#4a3aa7',
    'Kisumu': '#eda100',
    'Delhi': '#008300',
}

fig, ax = plt.subplots(figsize=(8, 7))
for location, color in location_colors.items():
    loc_data = rabbit_df[rabbit_df['location'] == location]
    ax.scatter(loc_data['umap_x'], loc_data['umap_y'], c=color, label=location,
               s=40, alpha=0.8, edgecolors='white', linewidth=0.5)

ax.set(xlabel='UMAP 1', ylabel='UMAP 2',
       title=f'UMAP of rabbit drawings (n={len(rabbit_df)}), colored by location')
ax.legend(title='Location')
plt.tight_layout()

output_dir = Path("../data/figures/umap_plots")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "umap_rabbit_by_location.png", dpi=150, bbox_inches='tight')
plt.show()
