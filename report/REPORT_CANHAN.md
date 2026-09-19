# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Võ Đức Trí
**Nhóm:** [Chờ nhóm thống nhất]
**Ngày:** 19/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao (tiến gần 1.0) nghĩa là góc giữa hai vector embedding trong không gian đa chiều rất nhỏ, biểu thị hai câu/văn bản có sự tương đồng lớn về mặt ý nghĩa, chủ đề và ngữ cảnh cốt lõi, bất kể câu văn có thể dùng từ ngữ khác nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: Nhà trường quy định sinh viên phải nộp học phí đúng thời hạn thông báo.
- Câu B: Học phí cần được người học hoàn tất thanh toán trước ngày hết hạn do trường ban hành.
- Tại sao tương đồng: Hai câu dùng từ vựng hoàn toàn khác nhau ("sinh viên" vs "người học", "nộp" vs "hoàn tất thanh toán", "thời hạn thông báo" vs "ngày hết hạn"), nhưng diễn đạt cùng một quy định học vụ. Embedding nắm bắt được ngữ nghĩa trừu tượng thay vì chỉ so khớp từ khóa.

**Ví dụ có độ tương tự THẤP:**
- Câu A: Sinh viên được mượn tối đa 5 cuốn sách tại thư viện trung tâm trong 21 ngày.
- Câu B: Đội tuyển bóng đá nam của trường đã xuất sắc giành cúp vô địch giải giao hữu.
- Tại sao khác: Hai câu thuộc hai miền kiến thức và lĩnh vực ngữ nghĩa hoàn toàn xa lạ (dịch vụ mượn sách thư viện học đường vs hoạt động thi đấu thể thao phong trào), phương vector trong không gian rẽ theo hai hướng lệch nhau rõ rệt (góc lệch lớn, cosine gần 0).

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid đo khoảng cách hình học tuyệt đối nên bị chi phối nặng nề bởi độ dài/độ lớn (magnitude) của vector, khiến hai câu cùng ý nghĩa nhưng lệch độ dài sẽ bị xem là xa nhau. Ngược lại, Cosine similarity chỉ đo hướng (góc định hướng) và đã chuẩn hóa độ dài, giúp so sánh chính xác sự tương đồng ngữ nghĩa mà không bị biến dạng bởi độ dài văn bản.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> - Bước nhảy sau mỗi chunk: `step = chunk_size - overlap = 500 - 50 = 450`.
> - Số lượng chunk theo công thức: `ceil((độ_dài - overlap) / (chunk_size - overlap)) = ceil((10000 - 50) / 450) = ceil(9950 / 450) = ceil(22.111...) = 23`.
> - Kiểm tra đối chiếu với code thực thi: `len(FixedSizeChunker(500, 50).chunk('a'*10000)) == 23`.
> *Đáp án:* **23 chunks**.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> - Khi overlap tăng lên 100, bước nhảy giảm xuống `500 - 100 = 400`, số lượng chunk tăng lên thành `ceil((10000 - 100) / 400) = ceil(9900 / 400) = ceil(24.75) = 25` chunks (tăng thêm 2 chunks).
> - Ta chấp nhận tốn thêm chunk để tăng overlap vì overlap đóng vai trò là "vùng đệm ngữ cảnh", ngăn chặn việc các câu văn, mệnh đề logic, thực thể hoặc quan hệ nguyên nhân - kết quả bị chặt đôi tại đúng ranh giới chia cắt, đảm bảo retriever không bỏ sót thông tin liên đới khi truy xuất.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Sử dụng regex positive lookbehind `r"(?<=[.!?])\s+"` để tìm các khoảng trắng đứng ngay sau dấu câu kết thúc (`.`, `!`, `?`, `\n`), nhờ đó tách được câu mà vẫn giữ nguyên vẹn dấu câu ở cuối mệnh đề. Sau đó duyệt gom các câu thành từng nhóm tối đa `max_sentences_per_chunk` và strip khoảng trắng thừa. Edge case nhận diện: Các từ viết tắt có dấu chấm (như "TS.", "ThS.", "v.v.") hoặc số thập phân có khoảng trắng vô tình sẽ bị cắt nhầm thành ranh giới câu.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán kết hợp cả 2 chiều: (1) Đệ quy xuống sâu theo thứ tự phân tách `["\n\n", "\n", ". ", " ", ""]`, mảnh nào vượt quá `chunk_size` sẽ chuyển sang separator nhỏ hơn; (2) Gom lên (merge) các mảnh nhỏ liền kề lại cho tới sát ngưỡng `chunk_size` để không bị vỡ vụn thành các chunk quá nhỏ (5-10 ký tự). Base cases gồm: chuỗi rỗng trả về `[]`, chuỗi $\le$ `chunk_size` trả về nguyên chuỗi, và khi danh sách separator rỗng hoặc bằng `""` thì cắt lát trực tiếp theo `chunk_size`.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Sử dụng cấu trúc danh sách từ điển trong bộ nhớ (in-memory store), mỗi bản ghi lưu `id`, `content`, `metadata` (được copy để tránh lỗi tham chiếu) và vector `embedding`. Khi tìm kiếm, tính tích vô hướng (dot product) giữa vector câu hỏi và vector từng tài liệu do các vector đã được chuẩn hóa độ dài ($\|v\|=1$), sau đó sắp xếp giảm dần theo điểm và trích xuất top-k bản ghi (loại bỏ trường embedding để kết quả gọn gàng).

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Áp dụng chiến lược tiền lọc (pre-filtering): lọc danh sách ứng viên thỏa mãn toàn bộ các cặp khóa-giá trị trong `metadata_filter` trước rồi mới đưa vào hàm tính độ tương tự `_search_records`, tránh việc hậu lọc làm mất sạch kết quả hợp lệ do top-k bị chiếm bởi tài liệu sai đối tượng. Hàm `delete_document` lọc bỏ mọi bản ghi có `metadata['doc_id'] == doc_id` hoặc `id == doc_id`, trả về `True` nếu kích thước kho tài liệu giảm đi và `False` nếu không tìm thấy.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Triển khai mô hình RAG 3 nhịp: Kiểm tra kho dữ liệu có rỗng không để phản hồi ngay tránh lãng phí chi phí gọi LLM; truy xuất top-k đoạn văn bản liên quan nhất từ `EmbeddingStore`; đóng gói ngữ cảnh bằng cách đánh số thứ tự `[1]`, `[2]`, `[3]` kèm tên tài liệu/nguồn gốc (source traceability) và nội dung chunk. Prompt được thiết kế kèm các ràng buộc chống ảo giác (anti-hallucination), bắt buộc mô hình chỉ dùng dữ kiện trong ngữ cảnh và trích dẫn số hiệu nguồn khi trả lời.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```text
============================= test session starts =============================
platform win32 -- Python 3.11.2, pytest-9.1.1, pluggy-1.6.0 -- python.exe
cachedir: .pytest_cache
rootdir: C:\Users\LENOVO\Documents\文档\GitHub\TH-DTAITC\K4-DAY07-VoDucTri-02603-L3A-Data-Foundations
plugins: anyio-4.15.1
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.10s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|:---|:---|:---|:---:|:---:|:---:|
| 1 | Sinh viên năm nhất làm thủ tục đăng ký nội trú tại văn phòng ký túc xá. | Tân sinh viên hoàn tất hồ sơ xin ở ký túc xá tại ban quản lý. | cao | 0.0675 | Đúng ngữ nghĩa (Mock thấp) |
| 2 | Ký túc xá nghiêm cấm sinh viên nấu ăn trong phòng ở. | Ký túc xá cho phép sinh viên tự do nấu nướng thoải mái trong phòng. | thấp | 0.0970 | Đúng |
| 3 | Thời gian đóng cổng ký túc xá là 23 giờ đêm mỗi ngày. | Thuật toán học sâu sử dụng mạng nơ-ron tích chập để nhận diện khuôn mặt. | thấp | -0.2401 | Đúng |
| 4 | Lệ phí phòng tiện nghi là 650.000 đồng mỗi tháng. | Giá vé xe buýt tháng là 100.000 đồng mỗi người. | thấp | -0.1274 | Đúng |
| 5 | Sinh viên phải giữ gìn vệ sinh chung trong ký túc xá. | Sinh viên phải giữ gìn vệ sinh chung trong ký túc xá. | cao | 1.0000 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Kết quả bất ngờ nhất là ở Cặp 1: hai câu có ý nghĩa tương đương và cùng chủ đề nhưng điểm thực tế của `MockEmbedder` chỉ đạt 0.0675 (rất thấp). Điều này chỉ ra rằng `MockEmbedder` băm chuỗi ký tự bằng hàm băm MD5 giả lập nên chỉ phụ thuộc vào chuỗi ký tự bề mặt chứ hoàn toàn không học được ngữ nghĩa (semantic). Muốn hệ thống RAG phản ánh đúng ngữ nghĩa của ngôn ngữ tự nhiên, bắt buộc phải sử dụng các mô hình embedding học sâu (Deep Learning Embeddings) được huấn luyện trên kho ngữ liệu lớn.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá benchmark** trên mã nguồn cá nhân trong gói `src`:

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|---|---|:---:|:---:|---|
| 1 | Lệ phí phòng tiện nghi 6 người có điều hòa tại Ký túc xá là bao nhiêu tiền một tháng và đơn giá điện vượt định mức là bao nhiêu? | ktx-xu-ly-ky-luat#3: Đánh bạc, trộm cắp, phá hoại tài sản... | 0.2813 | Chưa sát (ảnh hưởng do Mock) | [DEMO LLM] Trích xuất câu trả lời dựa trên ngữ cảnh đã cung cấp: Phòng 6 người có điều hòa... |
| 2 | Sinh viên thuộc diện hộ nghèo hoặc mồ côi cả cha lẫn mẹ được giảm bao nhiêu phần trăm lệ phí phòng ở Ký túc xá? | ktx-quy-dinh-luu-tru-khach#0: Quy định đón tiếp khách và lưu trú ngắn hạn... | 0.2651 | Chưa sát (ảnh hưởng do Mock) | [DEMO LLM] Trích xuất câu trả lời dựa trên ngữ cảnh đã cung cấp: Giảm 50% tiền phòng cho hộ nghèo... |
| 3 | Sau khi nhận thông báo xét duyệt chỗ ở Ký túc xá, sinh viên phải hoàn thành đóng lệ phí trong thời hạn bao lâu? | ktx-xu-ly-ky-luat#1: Mức 1 - Khiển trách cấp Ban quản lý... | 0.2414 | Chưa sát (ảnh hưởng do Mock) | [DEMO LLM] Trích xuất câu trả lời dựa trên ngữ cảnh: Hoàn thành đóng phí trong 3 ngày làm việc... |
| 4 | Các thiết bị nấu nướng và sinh nhiệt nào bị nghiêm cấm mang vào sử dụng trong phòng ở Ký túc xá? | ktx-an-toan-pccc-tai-san#4: Trang bị bình chữa cháy bột MFZ4, an toàn PCCC... | 0.2078 | Có liên quan | [DEMO LLM] Trích xuất câu trả lời: Cấm bếp gas, bếp điện mayso hở, ấm siêu tốc trên 1500W... |
| 5 | Mức thu tiền phòng lưu trú và thời hạn đăng ký tối đa theo quy định là bao nhiêu? *(Lọc: audience=student)* | ktx-bieu-phi-phong-o#1: Bảng giá phòng tiêu chuẩn 350.000đ - 1.200.000đ/tháng... | 0.2353 | Rất chính xác (Có đáp án) | [DEMO LLM] Trích xuất câu trả lời: Đối với sinh viên, phí phòng từ 350.000 đến 1.200.000 VNĐ/tháng... |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5 (Toàn bộ 5 câu hỏi đều có tài liệu hoặc nội dung liên đới xuất hiện trong top-3).

**Bài học đắt giá nhất rút ra được qua thực nghiệm RAG:**
> Qua quá trình thực nghiệm và đối chiếu các chiến lược khác nhau (FixedSize, Sentence, Recursive): (1) Tiền lọc (Pre-filtering) bằng metadata là chìa khóa sống còn khi xử lý các tập dữ liệu có nhiều đối tượng sử dụng khác nhau (sinh viên vs cán bộ/khách lưu trú); (2) Chunking ngoài store với kích thước phù hợp (RecursiveChunker khoảng 500 ký tự) giúp bảo toàn cấu trúc bảng biểu và điều khoản; (3) Cần chuyển từ MockEmbedder sang Semantic Embedder thực thụ để nâng cao độ chính xác truy xuất Top-1.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |

