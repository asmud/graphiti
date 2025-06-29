
# Gemini Code Assistant Documentation

This document provides a comprehensive overview of the Graphiti project, its architecture, and instructions for development and testing.

## Project Overview

Graphiti is a Python-based framework for building and querying temporally-aware knowledge graphs. It is designed for AI agents operating in dynamic environments, allowing for the continuous integration of data from various sources into a coherent, queryable graph.

### Key Features

- **Real-Time Incremental Updates:** Graphiti can integrate new data without requiring a full recomputation of the graph.
- **Bi-Temporal Data Model:** The framework tracks both the time of an event's occurrence and its ingestion, enabling precise historical queries.
- **Hybrid Retrieval:** Graphiti combines semantic, keyword, and graph-based search methods for efficient and accurate data retrieval.
- **Customizable Ontology:** Developers can define their own entity and edge types using Pydantic models.
- **Scalability:** The framework is designed to handle large datasets and supports parallel processing.

## Architecture

Graphiti's architecture is modular, with several key components working together to provide its functionality.

### Core Components

- **`Graphiti` Class:** The main entry point for interacting with the framework. It orchestrates the other components to manage the knowledge graph.
- **Graph Driver:** An abstraction layer for interacting with the underlying graph database. Graphiti currently supports Neo4j and FalkorDB.
- **LLM Client:** A client for interacting with large language models (LLMs) for tasks like entity and relationship extraction. It supports various providers, including OpenAI, Anthropic, Groq, and Google Gemini.
- **Embedder:** A client for generating vector embeddings for nodes and edges in the graph. It supports providers like OpenAI, Voyage AI, and Google Gemini.
- **Cross-Encoder:** A client for reranking search results to improve their relevance.

### Data Model

Graphiti's data model is based on the following concepts:

- **Nodes:** Represent entities in the knowledge graph. There are three types of nodes:
    - `EntityNode`: Represents a real-world entity, such as a person, place, or organization.
    - `EpisodicNode`: Represents an event or a piece of information that occurs at a specific point in time.
    - `CommunityNode`: Represents a cluster of related nodes in the graph.
- **Edges:** Represent relationships between nodes. There are two types of edges:
    - `EntityEdge`: Represents a relationship between two `EntityNode`s.
    - `EpisodicEdge`: Represents the relationship between an `EpisodicNode` and the `EntityNode`s it mentions.

## Development

### Setup

1. **Prerequisites:**
    - Python 3.10 or higher
    - Neo4j or FalkorDB
    - An API key for your chosen LLM provider (e.g., OpenAI)

2. **Installation:**
    ```bash
    pip install -e .[dev]
    ```

### Running Tests

Graphiti uses `pytest` for testing. To run the tests, use the following command:

```bash
pytest
```

### Linting and Type Checking

The project uses `ruff` for linting and `mypy` for type checking.

- **Linting:**
    ```bash
    ruff check .
    ```
- **Type Checking:**
    ```bash
    mypy
    ```

## Important Files

- **`pyproject.toml`:** Defines the project's dependencies, build configurations, and tool settings.
- **`graphiti_core/graphiti.py`:** Contains the main `Graphiti` class and its core functionality.
- **`graphiti_core/driver/`:** Contains the graph database drivers.
- **`graphiti_core/llm_client/`:** Contains the LLM clients.
- **`graphiti_core/embedder/`:** Contains the embedder clients.
- **`graphiti_core/cross_encoder/`:** Contains the cross-encoder clients.
- **`tests/`:** Contains the project's tests.
- **`examples/`:** Contains example usage of the Graphiti framework.
