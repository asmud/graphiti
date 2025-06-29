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

from sentence_transformers import SentenceTransformer

from graphiti_core.embedder.client import EmbedderClient


class LocalEmbedder(EmbedderClient):
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)

    async def create(self, input_data: list[str]) -> list[list[float]]:
        """
        Embeds a list of texts using a local sentence-transformers model.

        Parameters
        ----------
        input_data : list[str]
            The list of texts to embed.

        Returns
        -------
        list[list[float]]
            A list of embeddings.
        """
        return self.model.encode(input_data).tolist()
