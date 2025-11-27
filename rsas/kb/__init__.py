"""Knowledge Base module for RAG-based candidate search.

This module implements a three-tier search architecture:
- Tier 1: Metadata + Keyword Search (SQL + FTS5) - FREE
- Tier 2: Semantic Vector Search (ChromaDB) - CHEAP
- Tier 3: Deep Analysis with GPT-5.1 - EXPENSIVE

Cost optimization: 95-99% savings by filtering intelligently before expensive operations.
"""

__version__ = "0.1.0"
