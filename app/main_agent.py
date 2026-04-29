import uuid
import json

from langchain_core.messages import HumanMessage, AIMessage

# =========================================================
# AGENT GRAPH RUN
# =========================================================
def run_agent(
    user_query: str,
    thread_id: str | None = None,
    recursion_limit: int = 50,
    debug_messages: bool = False,
) -> str:
    """
    Chạy agent_graph với một câu hỏi, trả về chuỗi câu trả lời cuối cùng.

    Args:
        user_query:       Câu hỏi của người dùng.
        thread_id:        ID thread cho checkpointer (None = tạo mới).
        recursion_limit:  Giới hạn đệ quy của LangGraph.
        debug_messages:   In toàn bộ message history để debug.

    Returns:
        Câu trả lời cuối cùng dạng str.
    """

    from app.agents.graph import agent_graph  # noqa: E402

    config = {
        "configurable": {"thread_id": thread_id or str(uuid.uuid4())},
        "recursion_limit": recursion_limit,
    }

    result = agent_graph.invoke(
        {"messages": [HumanMessage(content=user_query)]},
        config=config,
    )

    # ---------- debug ----------
    if debug_messages:
        print("\n=== ALL MESSAGES ===")
        for i, m in enumerate(result.get("messages", []), 1):
            role = getattr(m, "type", m.__class__.__name__)
            content = getattr(m, "content", str(m))
            tool_calls = getattr(m, "tool_calls", None)
            print(f"\n[{i}] {role}:")
            if tool_calls:
                print(f"  tool_calls: {json.dumps(tool_calls, ensure_ascii=False)[:300]}")
            print(content[:2000])
    # ---------------------------

    # Lấy câu trả lời cuối cùng từ messages
    messages = result.get("messages", [])
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
            return m.content.strip()

    return "Không tìm thấy câu trả lời."


# =========================================================
# MAIN — test toàn bộ pipeline
# =========================================================
def main():
    # ── (Tuỳ chọn) Thêm PDF mới vào DB ──────────────────────────────────────
    # pdf_path = r"C:\RAG_PRO\data\xoai.pdf"
    # add_pdf_to_db(pdf_path)

    # user_query = "Tiêu chí đánh giá của phát hiện bất thường trên lá cà chua là gì"
    # user_query = "xoài là gì"
    user_query = "giá cổ phiếu FPT ngày 2026-04-28 là bao nhiêu"

    # # ── (A) Xem context_data thuần trước khi qua agent ───────────────────────
    # print("=" * 60)
    # print("(A) CONTEXT DATA TỪ HYBRID RETRIEVAL")
    # print("=" * 60)
    # context_data = query_with_context(query=user_query)
    # print("\n--- MAIN CONTEXT ---")
    # print(context_data["main_context"])
    # print("\n--- SECONDARY CONTEXT ---")
    # print(context_data["secondary_context"])
    # print("\n--- FINAL CONTEXT (send to LLM) ---")
    # print(context_data["final_context"])

    # ── (B) Chạy qua agent_graph đầy đủ ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("(B) AGENT GRAPH — FULL PIPELINE")
    print("=" * 60)

    test_config_thread = str(uuid.uuid4())

    # Import graph ở đây (sau khi llm + retriever đã sẵn sàng)
    from app.agents.graph import agent_graph  # noqa: E402

    config = {
        "configurable": {"thread_id": test_config_thread},
        "recursion_limit": 50,
    }

    result = agent_graph.invoke(
        {"messages": [HumanMessage(content=user_query)]},
        config=config,
    )

    print("\n=== FINAL STATE KEYS ===")
    print(list(result.keys()))

    print("\n=== FINAL ANSWER ===")
    messages = result.get("messages", [])
    final_answer = "Không tìm thấy câu trả lời."
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
            final_answer = m.content.strip()
            break
    print(final_answer)

    # (Tuỳ chọn) In toàn bộ message trace để debug luồng agent
    print("\n=== ALL MESSAGES (debug) ===")
    for i, m in enumerate(messages, 1):
        role = getattr(m, "type", m.__class__.__name__)
        content = getattr(m, "content", str(m))
        tool_calls = getattr(m, "tool_calls", None)
        print(f"\n[{i}] {role}:")
        if tool_calls:
            print(f"  tool_calls: {json.dumps(tool_calls, ensure_ascii=False)[:300]}")
        print(content[:2000])

if __name__ == "__main__":
    main()