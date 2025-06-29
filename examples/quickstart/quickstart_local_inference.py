"""
This example demonstrates how to use Graphiti with a local inference client (spaCy)
for knowledge graph extraction.
"""

import asyncio
from datetime import datetime

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.local_reranker import LocalReranker
from graphiti_core.embedder.local_embedder import LocalEmbedder
from graphiti_core.llm_client.local_llm_client import LocalLLMClient


async def main():
    """
    Main function to run the quickstart example.
    """
    # Initialize Graphiti with the LocalLLMClient, LocalEmbedder, and LocalReranker
    graphiti = Graphiti(
        "bolt://localhost:7687",
        "neo4j",
        "password",
        llm_client=LocalLLMClient(),
        embedder=LocalEmbedder(),
        cross_encoder=LocalReranker(),
    )

    # Build indices and constraints
    await graphiti.build_indices_and_constraints(delete_existing=True)

    # Add an episode to the graph
    await graphiti.add_episode(
        name="Kendra's Day",
        episode_body="Kendra went to the park. She saw a dog.",
        source_description="A summary of Kendra's day.",
        reference_time=datetime.now(),
    )

    # Search for relationships
    results = await graphiti.search("What did Kendra see?")

    # Print the results
    for edge in results:
        print(f"{edge.source.name} -> {edge.relationship} -> {edge.target.name}")

    # Close the connection
    await graphiti.close()


if __name__ == "__main__":
    asyncio.run(main())
