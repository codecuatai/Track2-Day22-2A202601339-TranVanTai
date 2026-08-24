"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import time
import types
import warnings
warnings.filterwarnings("ignore")

# Shim để khắc phục lỗi import vertexai trong langchain-community 0.4+
if "langchain_community.chat_models.vertexai" not in sys.modules:
    m = types.ModuleType("langchain_community.chat_models.vertexai")
    m.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = m

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
SYSTEM_V1 = (
    "Bạn là trợ lý AI hữu ích và ngắn gọn. Chỉ sử dụng context sau đây để trả lời câu hỏi. "
    "Giữ câu trả lời ngắn gọn, trực diện trong 2-3 câu. "
    "Nếu không tìm thấy thông tin trong context, hãy nói 'Tôi không tìm thấy thông tin này trong tài liệu.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "Bạn là một chuyên gia phân tích thông tin và kiến trúc sư AI. Hãy nghiên cứu kỹ context sau để trả lời: "
    "1) Tóm tắt trực tiếp ý chính, "
    "2) Trình bày chi tiết các luận điểm có tổ chức bằng gạch đầu dòng, "
    "3) Trích dẫn các sự kiện và thuật ngữ chính xác từ context. "
    "Nếu không đủ dữ liệu, hãy nêu rõ thông tin còn thiếu.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.
    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    """
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    ctx_str = "\n\n".join(contexts)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  ctx_str,
        "question": question,
    })

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        for attempt in range(4):
            try:
                out = run_rag(retriever, llm, prompt, qa["question"])
                results.append({
                    "question":  qa["question"],
                    "reference": qa["reference"],
                    "answer":    out["answer"],
                    "contexts":  out["contexts"],
                })
                print(f"  [{i:02d}/50] {qa['question'][:60]} -> {str(out['answer'])[:50]}...")
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"  ⚠️ Rate limit, tạm nghỉ 15s (lần thử {attempt + 1}/4)...")
                    time.sleep(15)
                else:
                    print(f"  ❌ Lỗi câu {i}: {e}")
                    results.append({
                        "question":  qa["question"],
                        "reference": qa["reference"],
                        "answer":    "Không tìm thấy thông tin phù hợp.",
                        "contexts":  [],
                    })
                    break

        time.sleep(1.0)

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=str(r["answer"]),
            retrieved_contexts=r["contexts"] if r["contexts"] else ["No context"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~3-5 phút)")

    dataset = build_ragas_dataset(rag_results)
    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
    )

    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]
        valid_vals = [v for v in raw if v is not None and not np.isnan(v)]
        scores[key] = float(np.mean(valid_vals)) if valid_vals else 0.0

    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    print("\n[1/4] Đang khởi tạo vectorstore...")
    vectorstore = setup_vectorstore()

    print("\n[2/4] Thu thập kết quả RAG cho cả V1 và V2...")
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    print("\n[3/4] Chạy RAGAS Evaluation...")
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    print("\n[4/4] Bảng so sánh & Xuất báo cáo JSON...")
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": bool(best_faith >= 0.8),
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n💾 Đã lưu báo cáo vào: {report_path}")
    print("✅ Hoàn thành Bước 3!")
    print("=" * 60)


if __name__ == "__main__":
    main()
