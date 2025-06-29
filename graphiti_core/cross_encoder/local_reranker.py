"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from sentence_transformers.cross_encoder import CrossEncoder

from graphiti_core.cross_encoder.client import CrossEncoderClient


class LocalReranker(CrossEncoderClient):
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model)

    async def rank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[dict]:
        """
        Not implemented for this client.
        """
        raise NotImplementedError(
            "This method is not implemented for the LocalReranker."
        )

    async def arerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[dict]:
        """
        Reranks a list of documents based on a query using a local cross-encoder model.

        Parameters
        ----------
        query : str
            The query to rerank against.
        documents : list[str]
            The list of documents to rerank.
        top_n : int | None, optional
            The number of documents to return. If None, returns all documents.

        Returns
        -------
        list[dict]
            A list of reranked documents with their scores.
        """
        if top_n is None:
            top_n = len(documents)

        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)

        reranked = [
            {"doc": doc, "score": score, "index": i}
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]
        reranked.sort(key=lambda x: x["score"], reverse=True)

        return reranked[:top_n]
