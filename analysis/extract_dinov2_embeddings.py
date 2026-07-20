# %%
# DINOv2 embedding extraction for the across-site drawings.
#
# Mirrors the embedding-extraction stage of embedding_retrieval.ipynb (cells 1-31),
# swapping the CLIP generator for DINOv2 (facebook/dinov2-base, CLS token, dim=768).
# Assumes the resized/cropped drawing directories already exist (produced by the
# "Resizing and cropping drawings" section of embedding_retrieval.ipynb) -- this
# script does not redo that step.
#
# DINOv2 has no text tower, so there is no text-embedding / recognizability step here.

from vislearnlabpy.embeddings.generate_embeddings import EmbeddingGenerator
from vislearnlabpy.embeddings.embedding_store import EmbeddingStore
from vislearnlabpy.embeddings.stimuli_loader import ImageExtractor, ImgExtractionSettings
import pandas as pd
from pathlib import Path

# %%
# same extraction settings as the CLIP pipeline in embedding_retrieval.ipynb --
# keeps the two embedding spaces comparable (same crop/resize going into the model)
dinov2_extraction_settings = ImgExtractionSettings(
    resize_dim=224,
    apply_content_crop=True,
    bg_component_size=0,
    apply_center_crop=False,
    use_thumbnail=False,
    filter_edge_artifacts=False,
    normalize_stroke_thickness=False
)
dinov2_transforms = ImageExtractor.get_transformations(dinov2_extraction_settings)

dinov2_generator = EmbeddingGenerator.from_model(
    "dinov2-base",
    device="mps",  # change to "cpu" or "cuda" as needed
    output_type="doc",
    transform=dinov2_transforms
)

# %%
drawings_folder = Path("/Volumes/vislearnlab/experiments/drawing/data")
beijing_resized_dir = drawings_folder / Path("beijing/resized_drawings")
kisumu_resized_dir = drawings_folder / Path("kisumu/resized_drawings")
newdelhi_dir = drawings_folder / Path("india/sketches_full_dataset")
newdelhi_df = pd.read_csv(drawings_folder / Path("india/AllDescriptives_images_final_india_run_v1.csv"))
newdelhi_subject_data = pd.read_csv(Path("/Volumes/vislearnlab/experiments/drawing/data/india/subject_data.csv"))

# %%
# Beijing + sanjose (same folder)
dinov2_generator.generate_image_embeddings(
    output_path="beijing_sanjose_drawings_resized_dinov2",
    input_dir=beijing_resized_dir,
    batch_size=100,
    overwrite=True
)

# %%
# Kisumu
dinov2_generator.generate_image_embeddings(
    output_path="kisumu_drawings_resized_dinov2",
    input_dir=kisumu_resized_dir,
    batch_size=100,
    overwrite=True
)

# %%
# Newdelhi -- same participant filtering as embedding_retrieval.ipynb cell 18
valid_pids = newdelhi_subject_data[newdelhi_subject_data["Age (months)"] > 0]["PID"].unique().tolist()
newdelhi_df = newdelhi_df.rename(columns={'filename': 'image1', 'category': 'text1'})
newdelhi_df['text1'] = newdelhi_df['text1'].apply(lambda x: " ".join(x.split('_')))
filtered_newdelhi_df = newdelhi_df[newdelhi_df['participant_id'].str.upper().isin(valid_pids)]
filtered_newdelhi_df.to_csv("tmp_newdelhi_draw_df.csv")

dinov2_generator.generate_image_embeddings(
    output_path="newdelhi_drawings_dinov2",
    input_csv="tmp_newdelhi_draw_df.csv",
    batch_size=100,
    overwrite=True
)

# %%
# Load the stores back in (model_type for dinov2-base preset is "dinov2-b")
newdelhi_store = EmbeddingStore.from_doc("newdelhi_drawings_dinov2/image_embeddings/dinov2-b_image_embeddings_doc")
kisumu_store = EmbeddingStore.from_doc("kisumu_drawings_resized_dinov2/image_embeddings/dinov2-b_image_embeddings_doc")
beijing_sanjose_store = EmbeddingStore.from_doc("beijing_sanjose_drawings_resized_dinov2/image_embeddings/dinov2-b_image_embeddings_doc")

# %%
# split beijing/sanjose the same way as embedding_retrieval.ipynb cell 21
# (only embeddings that include verbal cues, not picture cues)
# dim is explicit (768 for dinov2-base) since these stores are built via add_embedding
# rather than from_doc, and EmbeddingStore() otherwise defaults to CLIP's 512-dim schema
dinov2_dim = dinov2_generator.model.embedding_dim
sanjose_store = EmbeddingStore(dim=dinov2_dim)
beijing_store = EmbeddingStore(dim=dinov2_dim)
for embedding in beijing_sanjose_store.EmbeddingList:
    if "S_" in embedding.url:
        if "sanjose" in embedding.url:
            sanjose_store.add_embedding(embedding=embedding.embedding, url=embedding.url)
        else:
            beijing_store.add_embedding(embedding=embedding.embedding, url=embedding.url)

print(f"Kisumu embeddings: {len(kisumu_store.EmbeddingList)}")
print(f"Beijing embeddings: {len(beijing_store.EmbeddingList)}")
print(f"sanjose embeddings: {len(sanjose_store.EmbeddingList)}")
print(f"newdelhi embeddings: {len(newdelhi_store.EmbeddingList)}")

# %%
# each store above was built independently (two from_doc calls, two add_embedding-built),
# so their EmbeddingType classes are distinct dynamically-generated types even though
# structurally identical -- rebuild through a single store's add_embedding rather than
# concatenating DocLists of different types
full_embedding_store = EmbeddingStore(dim=dinov2_dim)
for store in [sanjose_store, beijing_store, kisumu_store, newdelhi_store]:
    for embedding in store.EmbeddingList:
        full_embedding_store.add_embedding(embedding=embedding.embedding, url=embedding.url)

# %%
# saving embedding files, alongside the CLIP stores with a _dinov2 suffix
output_embedding_dir = "../data/embeddings"
beijing_store.to_doc(f"{output_embedding_dir}/beijing_store_dinov2.doc")
kisumu_store.to_doc(f"{output_embedding_dir}/kisumu_store_dinov2.doc")
sanjose_store.to_doc(f"{output_embedding_dir}/sanjose_store_dinov2.doc")
newdelhi_store.to_doc(f"{output_embedding_dir}/newdelhi_store_dinov2.doc")
full_embedding_store.to_doc(f"{output_embedding_dir}/full_embedding_store_dinov2.doc")
