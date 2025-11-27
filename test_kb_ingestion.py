"""Test script for Knowledge Base ingestion pipeline.

This script tests the complete ingestion pipeline on existing candidates.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rsas.core.config.loader import load_config
from rsas.core.storage.object_store import ObjectStore
from rsas.integrations.openai_client import OpenAIClient
from rsas.kb.ingestion.pipeline import KnowledgeBaseIngestionPipeline
from rsas.kb.storage.chroma_client import get_chroma_client
from rsas.observability.logger import get_logger

logger = get_logger(__name__)


async def main():
    """Run ingestion pipeline test."""
    print("\n" + "=" * 60)
    print("KNOWLEDGE BASE INGESTION TEST")
    print("=" * 60)

    # 1. Load configuration
    print("\n[1/6] Loading configuration...")
    config = load_config()
    store_dir = config.get("storage", {}).get("object_store_dir", "data/processed")
    print(f"✓ Object store: {store_dir}")

    # 2. Initialize object store
    print("\n[2/6] Initializing object store...")
    store = ObjectStore(store_dir)
    print("✓ Object store ready")

    # 3. Initialize OpenAI client
    print("\n[3/6] Initializing OpenAI client...")
    openai_client = OpenAIClient(model="gpt-5.1")  # Use gpt-5.1 for summaries
    print("✓ OpenAI client ready")

    # 4. Initialize ChromaDB (in-memory for testing)
    print("\n[4/6] Initializing ChromaDB...")
    try:
        chroma_client = get_chroma_client(mode="memory")
        print(f"✓ ChromaDB ready (in-memory mode)")
        print(f"  Collection count: {chroma_client.count()}")
    except ImportError:
        print("⚠ ChromaDB not available - will skip vector storage")
        print("  Install with: pip install chromadb")
        chroma_client = None

    # 5. Create ingestion pipeline
    print("\n[5/6] Creating ingestion pipeline...")
    pipeline = KnowledgeBaseIngestionPipeline(
        store=store,
        openai_client=openai_client,
        chroma_client=chroma_client,
        skip_if_exists=True,  # Skip already ingested candidates
    )
    print("✓ Pipeline ready")

    # 6. Run ingestion
    print("\n[6/6] Running ingestion...")
    print("-" * 60)

    try:
        result = await pipeline.ingest_all_candidates(
            job_id=None,  # Ingest all candidates from all jobs
            batch_size=10,  # Small batch for testing
        )

        print("-" * 60)
        print("\nINGESTION RESULTS:")
        print(f"  ✓ Success: {result.success}")
        print(f"  📊 Candidates processed: {result.candidates_processed}")
        print(f"  ❌ Candidates failed: {result.candidates_failed}")
        print(f"  💰 Total cost: ${result.total_cost:.4f}")
        print(f"  🎫 Total tokens: {result.total_tokens:,}")
        print(f"  ⏱ Duration: {result.duration_seconds:.1f}s")

        if result.errors:
            print(f"\n⚠ Errors ({len(result.errors)}):")
            for error in result.errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(result.errors) > 5:
                print(f"  ... and {len(result.errors) - 5} more")

        # Show ChromaDB stats if available
        if chroma_client:
            count = chroma_client.count()
            print(f"\n📦 ChromaDB stats:")
            print(f"  Total embeddings: {count}")

    except Exception as e:
        print(f"\n❌ Ingestion failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60 + "\n")

    return 0 if result.success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
