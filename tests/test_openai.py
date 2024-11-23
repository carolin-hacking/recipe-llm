import numpy as np
from openai import OpenAI

from src.openai import get_embedding


def test_openai_key_works():
    """
    From https://platform.openai.com/docs/quickstart?language-preference=python
    Please set the env var OPENAI_API_KEY to your API key.
    """
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Write a haiku about recursion in programming.",
            },
        ],
    )

    message: str = completion.choices[0].message.content
    assert isinstance(message, str)


def test_embed_texts():
    texts = ["The dog barks while it runs!", "A cloud does its laundry in the sky."]
    embeds: np.ndarray = get_embedding(texts=texts)
    embed_dim = 1536  # see dims in https://platform.openai.com/docs/guides/embeddings
    assert embeds.shape == (len(texts), embed_dim)
