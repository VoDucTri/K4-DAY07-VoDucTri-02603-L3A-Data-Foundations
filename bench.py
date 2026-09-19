from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    LocalEmbedder,
    MockEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore


def parse_markdown_file(path: Path) -> tuple[dict[str, str], str]:
    """Parse frontmatter and body content from a markdown file."""
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) >= 3:
        frontmatter_text = parts[1]
        body = parts[2].strip()
        metadata = dict(
            re.findall(r"^(\w+):\s*(.+)$", frontmatter_text, flags=re.MULTILINE)
        )
        metadata = {k: v.strip().strip('"').strip("'") for k, v in metadata.items()}
    else:
        metadata = {}
        body = text.strip()

    metadata["doc_id"] = path.stem
    return metadata, body


def get_embedder():
    """Load configured embedding backend with fallback to MockEmbedder."""
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            return LocalEmbedder(
                model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL)
            )
        except Exception as err:
            print(f"Failed to load LocalEmbedder ({err}), falling back to MockEmbedder.")
            return MockEmbedder()
    elif provider == "openai":
        try:
            return OpenAIEmbedder(
                model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL)
            )
        except Exception as err:
            print(f"Failed to load OpenAIEmbedder ({err}), falling back to MockEmbedder.")
            return MockEmbedder()
    elif provider == "gemini":
        try:
            return GeminiEmbedder(
                model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL)
            )
        except Exception as err:
            print(f"Failed to load GeminiEmbedder ({err}), falling back to MockEmbedder.")
            return MockEmbedder()
    return MockEmbedder()


BENCHMARK_QUERIES = [
    {
        "id": "Q1",
        "type": "Tra cứu số liệu",
        "query": "Lệ phí phòng tiện nghi 6 người có điều hòa tại Ký túc xá là bao nhiêu tiền một tháng và đơn giá điện vượt định mức là bao nhiêu?",
        "gold_doc_id": "ktx-bieu-phi-phong-o",
        "gold_answer": "Phòng tiện nghi 6 người có điều hòa là 650.000 VNĐ/tháng/SV. Đơn giá điện vượt định mức là 2.500 VNĐ/kWh.",
        "keywords": ["650.000", "2.500"],
        "metadata_filter": None,
    },
    {
        "id": "Q2",
        "type": "Chính sách & Điều kiện",
        "query": "Sinh viên thuộc diện hộ nghèo hoặc mồ côi cả cha lẫn mẹ được giảm bao nhiêu phần trăm lệ phí phòng ở Ký túc xá?",
        "gold_doc_id": "ktx-chinh-sach-mien-giam",
        "gold_answer": "Sinh viên thuộc diện hộ nghèo và sinh viên mồ côi cả cha lẫn mẹ được giảm 50% lệ phí phòng ở.",
        "keywords": ["50%", "hộ nghèo"],
        "metadata_filter": None,
    },
    {
        "id": "Q3",
        "type": "Quy trình & Thủ tục",
        "query": "Sau khi nhận thông báo xét duyệt chỗ ở Ký túc xá, sinh viên phải hoàn thành đóng lệ phí trong thời hạn bao lâu?",
        "gold_doc_id": "ktx-thu-tuc-dang-ky",
        "gold_answer": "Sinh viên phải thanh toán tiền phòng và tiền thế chân trong vòng 3 ngày làm việc kể từ khi nhận thông báo.",
        "keywords": ["3 ngày làm việc"],
        "metadata_filter": None,
    },
    {
        "id": "Q4",
        "type": "An toàn & Danh mục cấm",
        "query": "Các thiết bị nấu nướng và sinh nhiệt nào bị nghiêm cấm mang vào sử dụng trong phòng ở Ký túc xá?",
        "gold_doc_id": "ktx-an-toan-pccc-tai-san",
        "gold_answer": "Nghiêm cấm mang vào phòng: bếp gas, bình gas mini, bếp củi, bếp than, bếp điện mayso hở, ấm đun/bàn là vượt quá 1500W.",
        "keywords": ["bếp gas", "1500W"],
        "metadata_filter": None,
    },
    {
        "id": "Q5",
        "type": "A/B Metadata Filter (Phân biệt đối tượng)",
        "query": "Mức thu tiền phòng lưu trú và thời hạn đăng ký tối đa theo quy định là bao nhiêu?",
        "gold_doc_id": "ktx-bieu-phi-phong-o",
        "gold_answer": "Đối với sinh viên: phòng tiêu chuẩn 350.000đ - 1.200.000đ/tháng, thu theo kỳ 5 tháng; còn đối với nhà khách/công tác: phòng đơn 300.000đ/ngày, lưu trú ngắn hạn tối đa 3 ngày.",
        "keywords": ["350.000", "tháng"],
        "metadata_filter": {"audience": "student"},
    },
]


def run_benchmark(data_dir: Path = Path("data/ky-tuc-xa")) -> str:
    lines = []
    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=" * 70)
    log("BENCHMARK RETRIEVAL — LAB 7: DATA FOUNDATIONS")
    log("Thành viên: Võ Đức Trí | Nhóm: [Chờ nhóm thống nhất] | Chủ đề: Ký túc xá")
    log("=" * 70)

    # 1. Đọc và nạp tài liệu
    md_files = sorted(data_dir.glob("*.md"))
    if not md_files:
        log(f"Lỗi: Không tìm thấy file .md trong {data_dir}")
        return "\n".join(lines)

    log(f"\n1. TẢI TÀI LIỆU TỪ: {data_dir} ({len(md_files)} files)")
    chunker = RecursiveChunker(chunk_size=500)
    all_chunks: list[Document] = []

    for file_path in md_files:
        metadata, body = parse_markdown_file(file_path)
        chunks = chunker.chunk(body)
        log(f"  - {file_path.name:32} -> {len(chunks)} chunks (audience: {metadata.get('audience', 'N/A')})")
        for idx, chunk_text in enumerate(chunks):
            doc = Document(
                id=f"{file_path.stem}#{idx}",
                content=chunk_text,
                metadata={**metadata, "doc_id": file_path.stem, "chunk_index": idx},
            )
            all_chunks.append(doc)

    log(f"\nTổng số chunks đã tạo: {len(all_chunks)}")

    # 2. Khởi tạo Embedder và Vector Store
    embedder = get_embedder()
    log(f"Backend Embedder: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="ktx_benchmark", embedding_fn=embedder)
    store.add_documents(all_chunks)
    log(f"Số Document nạp vào EmbeddingStore: {store.get_collection_size()}")

    # Demo agent
    agent = KnowledgeBaseAgent(store=store, llm_fn=lambda prompt: "Trích xuất câu trả lời dựa trên ngữ cảnh đã cung cấp.")

    # 3. Chạy 5 benchmark queries
    log("\n" + "=" * 70)
    log("2. KẾT QUẢ TRUY XUẤT 5 BENCHMARK QUERIES (TOP-3)")
    log("=" * 70)

    total_score = 0
    q_results = []

    for item in BENCHMARK_QUERIES:
        qid = item["id"]
        qtype = item["type"]
        qtext = item["query"]
        gold_id = item["gold_doc_id"]
        meta_filter = item["metadata_filter"]
        keywords = item["keywords"]

        log(f"\n--- [{qid}] ({qtype}) ---")
        log(f"Query: {qtext}")
        log(f"Tài liệu chuẩn (Gold Doc): {gold_id}")
        if meta_filter:
            log(f"Metadata Filter áp dụng: {meta_filter}")

        results = store.search_with_filter(qtext, top_k=3, metadata_filter=meta_filter)
        doc_in_top1 = False
        doc_in_top3 = False
        content_matched = False

        for rank, res in enumerate(results, start=1):
            res_doc_id = res.get("metadata", {}).get("doc_id", res.get("id"))
            res_chunk_id = res.get("id")
            score = res.get("score", 0.0)
            content = res.get("content", "")

            # Kiểm tra chứa từ khóa gold
            has_kw = any(kw.lower() in content.lower() for kw in keywords)
            is_gold = res_doc_id == gold_id

            if is_gold:
                doc_in_top3 = True
                if rank == 1:
                    doc_in_top1 = True
                if has_kw:
                    content_matched = True

            snippet = content[:120].replace("\n", " ")
            log(f"  Top-{rank} (score={score:.4f}): doc_id={res_doc_id} [{res_chunk_id}] {'[CHỨA ĐÁP ÁN]' if has_kw else ''}")
            log(f"         Preview: {snippet}...")

        # Đánh giá điểm theo rubric:
        # 2đ nếu gold ở top-1 và ngữ cảnh chứa đáp án; 1đ nếu gold ở top-2/3; 0đ nếu vắng
        score_point = 0
        if doc_in_top1 and content_matched:
            score_point = 2
            verdict = "XUẤT SẮC (Top-1 đúng tài liệu & chứa đáp án)"
        elif doc_in_top3:
            score_point = 1
            verdict = "ĐẠT (Nằm trong Top-3)"
        else:
            score_point = 0
            verdict = "CHƯA ĐẠT (Không nằm trong Top-3)"

        total_score += score_point
        log(f"-> Đánh giá: {verdict} ({score_point}/2 điểm)")
        q_results.append({
            "qid": qid,
            "qtext": qtext,
            "top1_doc": results[0]["metadata"]["doc_id"] if results else "N/A",
            "top1_score": results[0]["score"] if results else 0.0,
            "top1_content": results[0]["content"] if results else "",
            "relevant": "Có" if (doc_in_top1 or doc_in_top3) else "Không",
            "score_point": score_point,
        })

    # 4. Kiểm thử A/B bắt buộc cho Query 5 (Với filter vs Không filter)
    log("\n" + "=" * 70)
    log("3. KẾT QUẢ THỬ NGHIỆM A/B FILTERING (QUERY 5)")
    log("=" * 70)
    q5_text = BENCHMARK_QUERIES[4]["query"]
    log(f"Query: {q5_text}\n")

    log("[A] KHÔNG LỌC METADATA (metadata_filter=None):")
    res_no_filter = store.search_with_filter(q5_text, top_k=3, metadata_filter=None)
    for rank, r in enumerate(res_no_filter, start=1):
        doc_id = r["metadata"].get("doc_id")
        aud = r["metadata"].get("audience")
        log(f"  Top-{rank} (score={r['score']:.4f}): {doc_id} (audience: {aud})")

    log("\n[B] CÓ LỌC METADATA (metadata_filter={'audience': 'student'}):")
    res_filtered = store.search_with_filter(q5_text, top_k=3, metadata_filter={"audience": "student"})
    for rank, r in enumerate(res_filtered, start=1):
        doc_id = r["metadata"].get("doc_id")
        aud = r["metadata"].get("audience")
        log(f"  Top-{rank} (score={r['score']:.4f}): {doc_id} (audience: {aud})")

    log("\nNhận xét A/B:")
    log("- Khi không lọc, kết quả có thể lấy lẫn tài liệu nhà khách cán bộ/thân nhân (audience: faculty) hoặc nội quy chung.")
    log("- Khi kích hoạt metadata_filter={'audience': 'student'}, 100% tài liệu trả về được giới hạn chuẩn xác vào đúng đối tượng sinh viên nội trú.")

    log("\n" + "=" * 70)
    log(f"TỔNG KẾT: Điểm truy xuất = {total_score}/10")
    log("=" * 70)

    output_str = "\n".join(lines)
    Path("ket_qua_benchmark.txt").write_text(output_str, encoding="utf-8")
    log("\nĐã lưu kết quả chi tiết vào: ket_qua_benchmark.txt")
    return output_str


if __name__ == "__main__":
    run_benchmark()
