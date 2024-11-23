from openai import OpenAI
import numpy as np

client = OpenAI()


def get_embedding(texts: list[str], model="text-embedding-3-small") -> list[float]:
    texts = [preprocess(t) for t in texts]
    response = client.embeddings.create(input=texts, model=model)
    embeddings: list[list[float]] = [d.embedding for d in response.data]
    return np.array(embeddings)


def preprocess(text: str) -> str:
    return text.replace("\n", " ")
