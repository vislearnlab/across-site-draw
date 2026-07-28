# %%
# UMAP of rabbit drawings (CLIP embeddings) within the rabbit-only subspace,
# colored by age instead of location. Complementary to
# 03a_umap_rabbit_by_location.py -- same embedding/projection, different
# encoding, so the two plots can be compared side by side.
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
print(rabbit_df['age'].value_counts().sort_index())

# %%
embeddings = np.stack(rabbit_df['embedding'].values)

reducer = umap.UMAP(n_components=2, random_state=42)
coords = reducer.fit_transform(embeddings)
rabbit_df['umap_x'] = coords[:, 0]
rabbit_df['umap_y'] = coords[:, 1]

# %%
# ordinal light->dark blue ramp, one step per age (4-9) -- lightest step still
# clears the 2:1 contrast floor for ordinal (discrete-ordered) encoding
age_colors = {
    4: '#86b6ef',
    5: '#5598e7',
    6: '#2a78d6',
    7: '#1c5cab',
    8: '#104281',
    9: '#0d366b',
}

fig, ax = plt.subplots(figsize=(8, 7))
for age, color in age_colors.items():
    age_data = rabbit_df[rabbit_df['age'] == age]
    ax.scatter(age_data['umap_x'], age_data['umap_y'], c=color, label=age,
               s=40, alpha=0.8, edgecolors='white', linewidth=0.5)

ax.set(xlabel='UMAP 1', ylabel='UMAP 2',
       title=f'UMAP of rabbit drawings (n={len(rabbit_df)}), shaded by age')
ax.set_xticks([])
ax.set_yticks([])
ax.legend(title='Age')
plt.tight_layout()

output_dir = Path("../data/figures/umap_plots")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "umap_rabbit_by_age.png", dpi=150, bbox_inches='tight')
plt.show()