from pathlib import Path
import pandas as pd
import streamlit as st

st.title("Recipes")

data_dir = Path(__file__).parent / "data"
imgs_dir = data_dir / "recipe_images"
transcriptions_dir = data_dir / "responses" / "transcriptions"

transcriptions = [Path(f) for f in transcriptions_dir.glob("*.txt")]
tdf = pd.DataFrame({"path": transcriptions, "img": [f.name for f in transcriptions]})
st.write(tdf)

imgs = list(imgs_dir.glob("*.jpg"))
imgs_df = pd.DataFrame({"path": imgs, "img": [Path(f).name for f in imgs]})
event = st.dataframe(imgs_df, on_select="rerun", selection_mode="single-row")
if 0 == len(event["selection"]["rows"]):
    st.info("Please select a row")
    st.stop()

row: dict = imgs_df.loc[event["selection"]["rows"][0]].to_dict()
st.image(row["path"])
