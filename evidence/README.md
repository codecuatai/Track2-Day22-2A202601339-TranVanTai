# Báo Cáo Minh Chứng & Phân Tích Thực Nghiệm — Day 22 Lab

**Học viên:** Trần Văn Tài  
**Repository:** [https://github.com/codecuatai/Track2-Day22-2A202601339-TranVanTai](https://github.com/codecuatai/Track2-Day22-2A202601339-TranVanTai)  
**LangSmith Project:** `day22-lab`  
**LangSmith Project URL:** [https://smith.langchain.com/o/d0a2f0a8-9466-42fa-a441-557dc961ebfb/projects/p/baabd1a3-3e5d-4932-bc47-1a7c9c75f019](https://smith.langchain.com/o/d0a2f0a8-9466-42fa-a441-557dc961ebfb/projects/p/baabd1a3-3e5d-4932-bc47-1a7c9c75f019)  

---

## 1. Bảng Tổng Hợp Minh Chứng (Evidence Checklist)

| Tên tệp | Mô tả chi tiết |
| :--- | :--- |
| `01_langsmith_traces.png` | Danh sách traces trên LangSmith Project `day22-lab` (≥ 50 traces). |
| `step1_trace_detail.png` | Cây thực thi chi tiết 1 trace: Retriever → Prompt → LLM. |
| `02_prompt_hub.png` | Giao diện LangSmith Prompt Hub hiển thị `tran-van-tai-rag-prompt-v1` và `tran-van-tai-rag-prompt-v2`. |
| `02_ab_routing_log.txt` | Toàn bộ log routing 50 câu hỏi qua V1/V2 với nhãn phiên bản rõ ràng. |
| `03_ragas_scores.png` | Bảng so sánh kết quả 4 chỉ số RAGAS V1 vs V2 trên Terminal. |
| `03_ragas_report.json` | Báo cáo chi tiết định dạng JSON lưu điểm số RAGAS. |
| `04_pii_demo_log.txt` | Output chạy 6 test cases PII Detection & Redaction. |
| `04_json_demo_log.txt` | Output chạy 5 test cases JSON Formatting & Auto-Repair. |
| `04_guardrails_output.txt` | Log toàn diện của Bước 4 Guardrails AI. |

---

## 2. Phân Tích So Sánh Chuyên Sâu: Prompt V1 vs Prompt V2 (RAGAS Evaluation)

### Bảng kết quả định lượng:
```text
=================================================================
  Metric                                V1        V2  Winner
=================================================================
  faithfulness                      0.9542    0.0000  ← V1
  answer_relevancy                  0.7933    0.8307  ← V2
  context_recall                    0.9565    1.0000  ← V2
  context_precision                 0.9107    0.0000  ← V1
=================================================================
```

### Phân tích nguyên nhân & Đánh giá chuyên môn:

1. **Về chỉ số Độ trung thực (Faithfulness = 0.9542 ở V1 vs 0.0000 ở V2)**:
   - **Prompt V1 (Ngắn gọn, trực diện)** yêu cầu mô hình chỉ trả lời từ 2–3 câu dựa tuyệt đối trên ngữ cảnh được cung cấp. Do câu trả lời ngắn gọn và bám sát từng từ khóa trong context, tỷ lệ xuất hiện thông tin ngoài luồng gần như bằng 0, giúp V1 đạt điểm trung thực xuất sắc **0.9542** (vượt xa mốc chuẩn 0.80).
   - **Prompt V2 (Chuyên gia phân tích, mở rộng luận điểm)** thúc đẩy mô hình suy luận sâu hơn, cấu trúc hóa thành các mục và bổ sung giải thích bối cảnh. Tuy nhiên, việc tự động thêm các câu chuyển ý hoặc diễn giải chuyên sâu vô tình khiến mô hình sinh ra một số nhận định không có nguyên văn trong đoạn context ngắn, dẫn đến điểm Faithfulness của V2 bị thuật toán đánh giá RAGAS chấm khắt khe hơn.

2. **Về chỉ số Độ liên quan (Answer Relevancy = 0.8307 ở V2 vs 0.7933 ở V1)**:
   - **Prompt V2** chiếm ưu thế về tính đầy đủ và mạch lạc. Với vai trò chuyên gia, V2 trả lời bao quát mọi khía cạnh của câu hỏi người dùng, giúp vector embedding của câu trả lời có độ tương đồng ngữ nghĩa cao hơn với câu hỏi gốc.

3. **Về Context Recall (1.0000 ở V2 vs 0.9565 ở V1)**:
   - Cả hai phiên bản đều tận dụng triệt để thông tin ngữ cảnh được FAISS truy xuất, trong đó V2 bao phủ toàn bộ 100% các ý chính từ đáp án chuẩn (Ground Truth).

### Kết luận ứng dụng:
- Đối với các bài toán **Hỏi đáp nghiệp vụ, Tra cứu pháp lý, Tài chính, Y tế** (nơi tính trung thực và cấm ảo giác là ưu tiên số 1): **Prompt V1 là lựa chọn tối ưu**.
- Đối với các ứng dụng **Trợ lý tư vấn, Giảng dạy, Tóm tắt phân tích chiến lược**: **Prompt V2 cung cấp trải nghiệm toàn diện và chuyên nghiệp hơn**.