"""ChromaDB client wrapper for vector storage (Tier 2 search).

Provides in-memory or persistent vector storage for candidate embeddings.
Cost: $0 for storage, $0.000006 per query (OpenAI embedding only).
"""

from typing import Any

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from ...observability.logger import get_logger

logger = get_logger(__name__)


class ChromaClientWrapper:
    """Wrapper for ChromaDB client with candidate embeddings."""

    def __init__(
        self,
        mode: str = "memory",
        persist_directory: str | None = None,
        collection_name: str = "resume_summaries",
    ):
        """Initialize ChromaDB client.

        Args:
            mode: "memory" for in-memory or "persistent" for disk storage
            persist_directory: Directory for persistent storage
            collection_name: Name of the collection
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB not installed. Install with: pip install chromadb"
            )

        self.mode = mode
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

        # Initialize client
        if mode == "memory":
            self.client = chromadb.Client()
            self.logger.info("chroma_client_initialized", mode="in-memory")
        else:
            if not persist_directory:
                raise ValueError("persist_directory required for persistent mode")
            self.client = chromadb.Client(
                Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=persist_directory,
                )
            )
            self.logger.info(
                "chroma_client_initialized",
                mode="persistent",
                directory=persist_directory,
            )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        self.logger.info("collection_ready", name=collection_name)

    def add_embeddings(
        self,
        embeddings: list[list[float]],
        candidate_ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """Add embeddings to the collection.

        Args:
            embeddings: List of embedding vectors
            candidate_ids: List of candidate IDs (must be unique)
            metadatas: Optional list of metadata dicts
            documents: Optional list of document texts (summaries)
        """
        self.logger.info("adding_embeddings", count=len(embeddings))

        try:
            self.collection.add(
                embeddings=embeddings,
                ids=candidate_ids,
                metadatas=metadatas,
                documents=documents,
            )

            self.logger.info("embeddings_added", count=len(embeddings))

        except Exception as e:
            self.logger.error("add_embeddings_failed", error=str(e), exc_info=True)
            raise

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 20,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Query for similar candidates.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional metadata filters (e.g., {"candidate_id": {"$in": ["id1", "id2"]}})

        Returns:
            Dict with ids, distances, metadatas, documents
        """
        self.logger.info("querying_embeddings", n_results=n_results)

        try:
            kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
            }

            if where:
                kwargs["where"] = where

            results = self.collection.query(**kwargs)

            self.logger.info(
                "query_complete",
                results_count=len(results["ids"][0]) if results["ids"] else 0,
            )

            return results

        except Exception as e:
            self.logger.error("query_failed", error=str(e), exc_info=True)
            raise

    def get_by_ids(self, candidate_ids: list[str]) -> dict[str, Any]:
        """Get embeddings by candidate IDs.

        Args:
            candidate_ids: List of candidate IDs

        Returns:
            Dict with ids, embeddings, metadatas, documents
        """
        try:
            results = self.collection.get(ids=candidate_ids)
            return results
        except Exception as e:
            self.logger.error("get_by_ids_failed", error=str(e), exc_info=True)
            raise

    def delete_by_ids(self, candidate_ids: list[str]) -> None:
        """Delete embeddings by candidate IDs.

        Args:
            candidate_ids: List of candidate IDs to delete
        """
        try:
            self.collection.delete(ids=candidate_ids)
            self.logger.info("embeddings_deleted", count=len(candidate_ids))
        except Exception as e:
            self.logger.error("delete_failed", error=str(e), exc_info=True)
            raise

    def count(self) -> int:
        """Get total number of embeddings in collection.

        Returns:
            Count of embeddings
        """
        return self.collection.count()

    def clear(self) -> None:
        """Clear all embeddings from collection."""
        try:
            # Delete the collection and recreate
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self.logger.info("collection_cleared")
        except Exception as e:
            self.logger.error("clear_failed", error=str(e), exc_info=True)
            raise

    def persist(self) -> None:
        """Persist collection to disk (only for persistent mode)."""
        if self.mode == "persistent":
            try:
                self.client.persist()
                self.logger.info("collection_persisted")
            except Exception as e:
                self.logger.error("persist_failed", error=str(e), exc_info=True)
                raise


def get_chroma_client(
    mode: str = "memory",
    persist_directory: str | None = None,
    collection_name: str = "resume_summaries",
) -> ChromaClientWrapper:
    """Get ChromaDB client instance.

    Args:
        mode: "memory" or "persistent"
        persist_directory: Directory for persistent storage
        collection_name: Name of collection

    Returns:
        ChromaClientWrapper instance
    """
    return ChromaClientWrapper(
        mode=mode,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
