from __future__ import annotations

from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        """
        Retrieve top-k chunks, format context with source citations, and call LLM.
        """
        if self.store.get_collection_size() == 0:
            return "Không tìm thấy thông tin trong cơ sở dữ liệu (Kho tài liệu đang rỗng)."

        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy tài liệu phù hợp để trả lời câu hỏi."

        # Format context with numbering [1], [2], ... and document sources
        context_parts: list[str] = []
        for i, r in enumerate(results, start=1):
            source = (
                r.get("metadata", {}).get("title")
                or r.get("metadata", {}).get("doc_id")
                or r.get("id", f"doc_{i}")
            )
            context_parts.append(f"[{i}] Nguồn ({source}):\n{r['content']}")

        context = "\n\n".join(context_parts)
        prompt = (
            "Bạn là trợ lý hỏi đáp thông minh dựa trên tri thức được cung cấp.\n"
            "Hãy trả lời câu hỏi dưới đây chỉ dựa trên ngữ cảnh được cung cấp.\n"
            "Ràng buộc:\n"
            "- Không tự bịa đặt hoặc suy đoán thông tin ngoài ngữ cảnh.\n"
            "- Nếu thông tin không có trong ngữ cảnh, hãy nêu rõ là không tìm thấy.\n"
            "- Trích dẫn số hiệu nguồn [1], [2] khi đưa ra các dữ kiện.\n\n"
            f"Ngữ cảnh:\n{context}\n\n"
            f"Câu hỏi: {question}\n\n"
            "Câu trả lời:"
        )
        return self.llm_fn(prompt)
