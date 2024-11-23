import json
from pathlib import Path
from typing import Iterable

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

    def get_closest(self, to_idx: int) -> Iterable[tuple[int, float]]:
        assert 0 <= to_idx
        assert to_idx < len(self.emebds)
        single_emb = self.emebds[to_idx]
        similarities = self.emebds @ single_emb
        most_similar = rank_similarities(similarities)
        for rank in most_similar:
            if rank == to_idx:
                continue
            yield rank, similarities[rank]


def rank_similarities(sims: np.ndarray) -> np.ndarray:
    return np.argsort(sims)[::-1]
