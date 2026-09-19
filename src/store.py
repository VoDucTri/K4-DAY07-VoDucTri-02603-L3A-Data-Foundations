from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Always uses an in-memory store for consistent execution across test and production environments.
    The embedding_fn parameter allows injection of mock or real embeddings.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False  # Explicitly disabled to use in-memory store
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Normalize a Document into a stored record."""
        metadata = dict(doc.metadata) if doc.metadata else {}
        # Ensure doc_id exists in metadata, defaulting to doc.id
        if "doc_id" not in metadata:
            metadata["doc_id"] = doc.id
        embedding = self._embedding_fn(doc.content)
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": embedding,
        }

    def _search_records(
        self, query: str, records: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """Run in-memory similarity search over a provided list of candidate records."""
        if not records or top_k <= 0:
            return []

        query_embedding = self._embedding_fn(query)
        scored: list[dict[str, Any]] = []

        for record in records:
            doc_embedding = record["embedding"]
            score = _dot(query_embedding, doc_embedding)
            # Do not include the large embedding vector in the output dictionary
            scored.append(
                {
                    "id": record["id"],
                    "content": record["content"],
                    "metadata": record["metadata"],
                    "score": score,
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """Embed each document's content and store it in memory."""
        for doc in docs:
            record = self._make_record(doc)
            self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Find the top_k most similar documents to the query."""
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(
        self, query: str, top_k: int = 3, metadata_filter: dict | None = None
    ) -> list[dict[str, Any]]:
        """
        Search with optional metadata pre-filtering.

        First filter candidate chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            return self._search_records(query, self._store, top_k)

        candidates = [
            record
            for record in self._store
            if all(
                record.get("metadata", {}).get(k) == v
                for k, v in metadata_filter.items()
            )
        ]
        return self._search_records(query, candidates, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document (where metadata['doc_id'] == doc_id or id == doc_id).

        Returns True if any chunks were removed, False otherwise.
        """
        initial_size = len(self._store)
        self._store = [
            record
            for record in self._store
            if record.get("metadata", {}).get("doc_id") != doc_id
            and record.get("id") != doc_id
        ]
        return len(self._store) < initial_size
