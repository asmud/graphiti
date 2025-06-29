"""
Unit tests for the local clients.
"""

import pytest
from graphiti_core.cross_encoder.local_reranker import LocalReranker
from graphiti_core.embedder.local_embedder import LocalEmbedder
from graphiti_core.llm_client.local_llm_client import LocalLLMClient
from graphiti_core.prompts.extract_nodes import ExtractedEntities
from graphiti_core.prompts.models import Message


@pytest.mark.asyncio
async def test_local_llm_client():
    """Tests the LocalLLMClient."""
    client = LocalLLMClient()
    prompt = [
        Message(
            role="user",
            content="<CURRENT MESSAGE>\nKendra went to the park. She saw a dog.\n</CURRENT MESSAGE>",
        )
    ]
    response = await client._generate_response(
        prompt,
        response_model=ExtractedEntities,
        string_to_extract_from="Kendra went to the park. She saw a dog.",
    )
    assert "extracted_entities" in response
    assert len(response["extracted_entities"]) > 0
    assert response["extracted_entities"][0]["name"] == "Kendra"



@pytest.mark.asyncio
async def test_local_embedder():
    """Tests the LocalEmbedder."""
    embedder = LocalEmbedder()
    embeddings = await embedder.create(input_data=["hello world"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384  # all-MiniLM-L6-v2 embedding dimension


@pytest.mark.asyncio
async def test_local_reranker():
    """Tests the LocalReranker."""
    reranker = LocalReranker()
    query = "What is the capital of France?"
    documents = ["Paris is the capital of France.", "The sky is blue."]
    reranked = await reranker.arerank(query, documents)
    assert len(reranked) == 2
    assert reranked[0]["doc"] == "Paris is the capital of France."

