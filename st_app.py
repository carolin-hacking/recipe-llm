from pathlib import Path
import pandas as pd
import streamlit as st

from src.embeddings import Embeddings

st.set_page_config(layout="wide")


@st.cache_resource
def embeddings_container() -> Embeddings:
    return Embeddings(dir="01")


def get_images_folder_name(transcription_path: str) -> str:
    return f"data/recipe_images/{str(transcription_path).split('_')[1].split('.')[0]}"


st.title("Recipes")

data_dir = Path(__file__).parent / "data"
imgs_dir = data_dir / "recipe_images"
transcriptions_dir = data_dir / "responses" / "transcriptions"

transcription_files = [str(f) for f in transcriptions_dir.glob("*.txt")]
tdf = (
    pd.DataFrame(
        {
            "Transcription path": transcription_files,
            "Folder with images": [
                get_images_folder_name(f) for f in transcription_files
            ],
        }
    )
    .assign(recipe_idx=lambda df: df["Folder with images"].str[-1].astype(int))
    .sort_values("recipe_idx")
    .reset_index(drop=True)
)

event = st.dataframe(tdf, on_select="rerun", selection_mode="single-row")
if 0 == len(event["selection"]["rows"]):
    st.info("Please select a row")
    st.stop()

row: dict = tdf.loc[event["selection"]["rows"][0]].to_dict()

col1, col2, col3 = st.columns(3)
with col1:
    for img_path in Path(row["Folder with images"]).glob("*.jpg"):
        st.text(img_path.name)
        st.image(str(img_path))
with col2:
    st.write(Path(row["Transcription path"]).read_text())
with col3:
    st.write("## Most Similar Recipes")
    st.write("Top 10")
    emb = embeddings_container()
    edf = pd.DataFrame(
        list(emb.get_closest(to_idx=row["recipe_idx"])),
        columns=["Recipe idx", "Similarity"],
    )
    st.dataframe(edf.head(10), hide_index=True)
