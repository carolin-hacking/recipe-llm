from pathlib import Path
import pandas as pd
import streamlit as st


def get_images_folder_name(transcription_path: str) -> str:
    return f"data/recipe_images/{str(transcription_path).split('_')[1].split('.')[0]}"


st.title("Recipes")

data_dir = Path(__file__).parent / "data"
imgs_dir = data_dir / "recipe_images"
transcriptions_dir = data_dir / "responses" / "transcriptions"

transcriptions = [str(f) for f in transcriptions_dir.glob("*.txt")]
tdf = pd.DataFrame({"Transcription path": transcriptions, "images_folder": [get_images_folder_name(f) for f in transcriptions]})
event = st.dataframe(tdf, on_select="rerun", selection_mode="single-row")

if 0 == len(event["selection"]["rows"]):
    st.info("Please select a row")
    st.stop()

row: dict = tdf.loc[event["selection"]["rows"][0]].to_dict()

col1, col2 = st.columns(2)
with col1:
    for img_path in Path(row["images_folder"]).glob("*.jpg"):
        st.text(img_path.name)
        st.image(str(img_path))
with col2:
    st.text(Path(row["Transcription path"]).read_text())
