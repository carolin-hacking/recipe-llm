import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.openai_interaction import get_embedding


data_dir = Path(__file__).parent.parent / "data"
recipe_files = data_dir / "responses" / "transcriptions"
assert recipe_files.exists(), recipe_files.absolute()

transcription_files = [Path(f) for f in recipe_files.glob("*.txt")]
tdf = (
    pd.DataFrame(
        {"path": transcription_files, "filename": [f.name for f in transcription_files]}
    )
    .assign(
        recipe=lambda x: x.filename.str.split("_")
        .str[1]
        .str.split(".")
        .str[0]
        .astype(int)
    )
    .sort_values("recipe")
)

texts = [f.read_text() for f in transcription_files]
embeds: np.ndarray = get_embedding(texts)

# Save embeddings
tgt_dir = data_dir / "embeddings" / "01"
tgt_dir.mkdir(parents=True, exist_ok=True)

embed_file = tgt_dir / "embeds.npy"
np.save(file=embed_file, arr=embeds)
print("Saved embeddings to %s" % embed_file)

# Save metadata
metadata = {
    "time": datetime.datetime.now(),
    "shape": embeds.shape,
    "dtype": str(embeds.dtype),
}
description_file = tgt_dir / "description.json"
description_file.write_text(json.dumps(metadata, indent=2, default=str))
print("Saved embeddings metadata to %s" % embed_file)
