from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List

class SentenceEncoder(SentenceTransformer):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: List[str]):
        return self.model.encode(text)

    def find_best_index(self, query: int, corpus):
        """
        Consider making the distance function a parameter, if useful,
            so you can use other like L2, etc. instead of just cos_sim or dot_score.
        :param query: age given.
        :param corpus: .txt file we're using.
        :return: best index according to the transformer.
        """
        query_age = str(query)
        query_embedding = self.embed([query_age])
        corpus_embedding = self.embed(corpus)
        similarities = util.cos_sim(query_embedding, corpus_embedding)[0]

        best_idx = int(similarities.argmax())
        return best_idx
