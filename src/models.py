from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List

class SentenceEncoder():
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: List[str]):
        return self.model.encode(text)

    def find_best_index(self, query, corpus_embedding):
        """
        Consider making the distance function a parameter, if useful,
            so you can use other like L2, etc. instead of just cos_sim or dot_score.
        :param query: age given.
        :param corpus_embedding: Embedded dataset
        :return: best index according to the transformer.
        """
        query_age = str(query)
        query_embedding = self.embed([query_age])
        similarities = util.cos_sim(query_embedding, corpus_embedding)[0]

        best_idx = int(similarities.argmax())
        return best_idx

from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class GeminiEncoder():
    def __init__(self, model_name: str = "gemini-embedding-001"):
        self.model = model_name
        self.client = genai.Client(api_key='AIzaSyCHY6HWu2QwM-D6T9xBvW0--FuXziWvky4')

    def embed(self, sent):
        result = self.client.models.embed_content(model=self.model,
                                                  contents=sent,
                                                  config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"))
        return np.vstack([np.array(e.values, dtype=np.float32) for e in result.embeddings])

    def find_best_index(self, query_emb, emb_corpus):
        sims = cosine_similarity(query_emb, emb_corpus)

        best_idx = int(np.argmax(sims))
        return best_idx