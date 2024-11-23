import json
from pathlib import Path

import numpy as np


embeddings_dir = Path(__file__).parent.parent / "data" / "embeddings"


class Embeddings:
    def __init__(self, dir: str):
        folder = embeddings_dir / dir
        assert folder.exists(), folder.absolute()
        self.emebds = np.load(folder / "embeds.npy")
        self.description = json.loads((folder / "description.json").read_text())

    def get_embeds(self, index: int) -> np.ndarray:
        return self.emebds[index]

    def get_closest(self, to_idx: int) -> list[int]:
        assert 0 <= to_idx
        assert to_idx < len(self.emebds)
        single_emb = self.emebds[to_idx]
        similarities = self.emebds @ single_emb
        ranks = rank_similarities(similarities)
        return [r for r in ranks if r != to_idx]


def rank_similarities(sims: np.ndarray) -> np.ndarray:
    return np.argsort(sims)[::-1]
