"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com
"""
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
PROMPT_V1_NAME = "tran-van-tai-rag-prompt-v1"
PROMPT_V2_NAME = "tran-van-tai-rag-prompt-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
# V1: Phong cách ngắn gọn, xúc tích, trả lời trực diện 2-3 câu
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

# V2: Phong cách chuyên gia AI, phân tích chi tiết, có tổ chức và trích dẫn context
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


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def push_prompts_to_hub(client: Client):
    """
    Upload cả 2 prompt templates lên LangSmith Prompt Hub.
    """
    try:
        url_v1 = client.push_prompt(
            PROMPT_V1_NAME,
            object=PROMPT_V1,
            description="V1: Phong cách ngắn gọn, xúc tích (2-3 câu)"
        )
        print(f"✅ Đã push V1 → {url_v1}")
    except Exception as e:
        print(f"⚠️  V1 lỗi: {e}")

    try:
        url_v2 = client.push_prompt(
            PROMPT_V2_NAME,
            object=PROMPT_V2,
            description="V2: Phong cách chuyên gia, phân tích có cấu trúc"
        )
        print(f"✅ Đã push V2 → {url_v2}")
    except Exception as e:
        print(f"⚠️  V2 lỗi: {e}")


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """
    Tải 2 prompt từ LangSmith Prompt Hub.
    Fallback về template local nếu Hub không khả dụng.
    """
    prompts = {}

    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Đã pull '{PROMPT_V1_NAME}' từ Hub")
    except Exception as e:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V1_NAME}': {e}")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Đã pull '{PROMPT_V2_NAME}' từ Hub")
    except Exception as e:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V2_NAME}': {e}")

    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Xác định prompt version dựa trên MD5 hash của request_id.
    Quy tắc: hash chẵn → PROMPT_V1_NAME | hash lẻ → PROMPT_V2_NAME
    TÍNH CHẤT: cùng request_id LUÔN cho cùng kết quả (deterministic).
    """
    hash_int = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Chạy RAG chain với prompt version được chọn bởi router.
    """
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return {
        "question": question,
        "answer": answer,
        "version": version,
    }


# ── 7. Setup Vectorstore (tái sử dụng logic Bước 1) ───────────────────────
def setup_vectorstore():
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
def main():
    import time
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    client = Client(api_key=config.LANGSMITH_API_KEY)

    print("\n[1/4] Đang push prompts lên LangSmith Prompt Hub...")
    push_prompts_to_hub(client)

    print("\n[2/4] Đang pull prompts từ LangSmith Prompt Hub...")
    prompts = pull_prompts_from_hub(client)

    print("\n[3/4] Đang khởi tạo vectorstore và retriever...")
    vectorstore = setup_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = get_llm()

    log_lines = []
    header_log = f"=== A/B Routing Log ({len(SAMPLE_QUESTIONS)} queries) ==="
    print("\n" + header_log)
    log_lines.append(header_log)

    v1_count, v2_count = 0, 0
    print("\n[4/4] Đang thực thi A/B routing qua 50 câu hỏi...")

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        request_id  = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        try:
            result = ask_ab(retriever, llm, prompt, question, version_tag)
            log_entry = f"[{i:02d}] [{request_id}] [prompt-{version_tag}] Q: {question[:50]}... -> A: {str(result['answer'])[:60]}..."
            print(log_entry)
            log_lines.append(log_entry)
        except Exception as e:
            log_entry = f"[{i:02d}] [{request_id}] [prompt-{version_tag}] ❌ Lỗi: {e}"
            print(log_entry)
            log_lines.append(log_entry)

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1

        time.sleep(1.2)

    summary_log = f"\n📊 Thống kê Routing: V1={v1_count} câu | V2={v2_count} câu | Tổng={len(SAMPLE_QUESTIONS)}"
    print(summary_log)
    log_lines.append(summary_log)

    evidence_log_path = Path(__file__).parent.parent / "evidence" / "02_ab_routing_log.txt"
    evidence_log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"📝 Đã lưu log minh chứng vào: {evidence_log_path}")

    print("\n✅ Bước 2 hoàn thành! Hãy mở LangSmith Prompt Hub để kiểm tra và chụp ảnh minh chứng.")
    print("=" * 60)


if __name__ == "__main__":
    main()
