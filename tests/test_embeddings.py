import numpy as np
from src.embeddings import Embeddings
from src.embeddings import rank_similarities


def test_embeddings_container():
    e = Embeddings(dir="01")
    assert isinstance(e.description, dict)

    assert isinstance(e.get_embeds(0), np.ndarray)

    closest = e.get_closest(to_idx=0)
    assert isinstance(closest, list)
    assert closest[0] != 0


def test_rank_similarity():
    similarities = np.array([0.1, 0.9, 0.8])
    ranks = np.array([1, 2, 0])
    assert np.allclose(ranks, rank_similarities(similarities))


