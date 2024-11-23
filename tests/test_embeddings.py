import numpy as np
from src.embeddings import Embeddings
from src.embeddings import rank_similarities


def test_embeddings_container():
    e = Embeddings(dir="01")
    assert isinstance(e.description, dict)

    assert isinstance(e.get_embeds(0), np.ndarray)

    (closest, score), *_ = e.get_closest(to_idx=0)
    assert closest != 0
    assert 0 < score < 1


def test_rank_similarity():
    similarities = np.array([0.1, 0.9, 0.8])
    most_similar = np.array([1, 2, 0])
    assert np.allclose(most_similar, rank_similarities(similarities))
