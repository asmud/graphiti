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

import json
from typing import Any, Coroutine, Type

import spacy
from pydantic import BaseModel

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.prompts.extract_edges import ExtractedEdges
from graphiti_core.prompts.extract_nodes import ExtractedEntities


class LocalLLMClient(LLMClient):
    def __init__(self, model: str = "xx_sent_ud_sm"):
        super().__init__(LLMConfig())  # Call parent constructor
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"Downloading spaCy model {model}...")
            spacy.cli.download(model)
            self.nlp = spacy.load(model)

    async def _generate_response(
        self,
        prompt: list[Any],
        response_model: Type[BaseModel] | None = None,
        model_size: ModelSize = ModelSize.small,
        max_tokens: int | None = None,
        string_to_extract_from: str | None = None,
    ) -> dict:
        if not string_to_extract_from:
            return {}

        doc = self.nlp(string_to_extract_from)

        if response_model == ExtractedEntities:
            nodes = [{"name": ent.text, "entity_type_id": 0} for ent in doc.ents]
            return {"extracted_entities": nodes}

        if response_model == ExtractedEdges:
            # This is a placeholder for a more robust node extraction from the prompt
            nodes_from_prompt = []
            user_prompt = next((p.content for p in prompt if p.role == "user"), "")
            try:
                nodes_str = user_prompt.split("<NODES>")[1].split("</NODES>")[0]
                nodes_from_prompt = json.loads(nodes_str)
            except (IndexError, json.JSONDecodeError):
                pass

            edges = []
            for token in doc:
                if token.dep_ == "dobj" and token.head.pos_ == "VERB":
                    subjs = [w for w in token.head.lefts if w.dep_ == "nsubj"]
                    if subjs:
                        source_text = subjs[0].text
                        target_text = token.text
                        relation = token.head.text

                        source_id = next((n['id'] for n in nodes_from_prompt if n['name'] == source_text), -1)
                        target_id = next((n['id'] for n in nodes_from_prompt if n['name'] == target_text), -1)

                        if source_id != -1 and target_id != -1:
                            edges.append({
                                "source_entity_id": source_id,
                                "target_entity_id": target_id,
                                "fact": f"{source_text} {relation} {target_text}",
                                "relation_type": relation.upper(),
                            })
            return {"edges": edges}

        return {}

    def acreate_structured_llm_completion(
        self,
        prompt: str,
        string_to_extract_from: str,
        structured_response_model: Type[BaseModel],
    ) -> Coroutine[Any, Any, BaseModel]:
        raise NotImplementedError("Use generate_response for LocalLLMClient")

    async def acreate_llm_completion(
        self,
        prompt: str,
        string_to_extract_from: str,
    ) -> str:
        raise NotImplementedError("This method is not implemented for the LocalLLMClient.")
