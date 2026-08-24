"""
Bước 1 — RAG Pipeline với LangSmith Tracing
=============================================
NHIỆM VỤ:
  1. Tải knowledge base, chia chunks, index với FAISS
  2. Xây dựng RAG chain: retriever → prompt → LLM → output parser
  3. Trang trí hàm query với @traceable để LangSmith ghi lại mỗi lần gọi
  4. Chạy 50 câu hỏi → tạo ≥ 50 traces trên LangSmith

DELIVERABLE: Mở https://smith.langchain.com → project của bạn → xác nhận ≥ 50 traces.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ⚠️ QUAN TRỌNG: Import config TRƯỚC KHI import bất kỳ thư viện LangChain nào.
# config.py tự động đặt LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, ... vào os.environ
import config

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable, Client

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Thiết lập Vectorstore ───────────────────────────────────────────────
def setup_vectorstore():
    """
    Tải knowledge base, chia chunks và tạo FAISS vectorstore.
    """
    embeddings = get_embeddings()
    text = load_knowledge_base()
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    print(f"📚 Đã chia thành {len(chunks)} chunks")
    vectorstore = build_vectorstore(chunks, embeddings)
    return vectorstore


# ── 2. RAG Prompt Template ─────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Bạn là trợ lý AI hữu ích. Chỉ dùng context sau để trả lời.\n\nContext:\n{context}"),
    ("human", "{question}"),
])


# ── 3. Build RAG Chain ─────────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """
    Xây dựng LCEL RAG chain theo cấu trúc pipe:
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()

    Trả về: (chain, retriever)
    """
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# ── 4. Hàm Query có LangSmith Tracing ─────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """
    Chạy RAG chain với một câu hỏi.
    Decorator @traceable sẽ gửi mỗi lần gọi lên LangSmith như một trace riêng.
    """
    return chain.invoke(question)


# ── 5. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 1: LangSmith RAG Pipeline")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    print("\n[1/3] Đang khởi tạo vectorstore...")
    vectorstore = setup_vectorstore()

    print("\n[2/3] Đang xây dựng RAG chain...")
    chain, retriever = build_rag_chain(vectorstore)

    print(f"\n[3/3] Đang thực thi {len(SAMPLE_QUESTIONS)} câu hỏi qua RAG pipeline...")
    start_time = time.time()

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        q_start = time.time()
        try:
            answer = ask(chain, question)
            q_latency = time.time() - q_start
            print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] ({q_latency:.2f}s) Q: {question[:60]}")
            print(f"       A: {str(answer)[:90]}...\n")
        except Exception as e:
            print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] ❌ Lỗi: {e}\n")

        time.sleep(1.2)

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"⏱️ Tổng thời gian thực hiện: {total_time:.2f}s")
    print(f"✅ {len(SAMPLE_QUESTIONS)} queries đã hoàn thành và gửi lên LangSmith project '{config.LANGSMITH_PROJECT}'")

    try:
        client = Client(api_key=config.LANGSMITH_API_KEY)
        runs = list(client.list_runs(project_name=config.LANGSMITH_PROJECT, execution_order=1, limit=100))
        print(f"🔍 [LangSmith Verification] Xác nhận: Hiện có {len(runs)} traces trong project '{config.LANGSMITH_PROJECT}' (Yêu cầu Rubric: >= 50).")
    except Exception as e:
        print(f"ℹ️ Kiểm tra traces qua web dashboard: {e}")

    print("\n🌐 Mở https://smith.langchain.com để xem chi tiết traces và chụp ảnh lưu vào thư mục evidence/")
    print("=" * 60)


if __name__ == "__main__":
    main()
