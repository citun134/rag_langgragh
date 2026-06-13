import re
import json
import uuid
from typing import Literal
from langgraph.types import Send

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    RemoveMessage,
    ToolMessage,
)

from app.agents.state import State, AgentState, QueryAnalysis
from app.config.prompts import *

from app.agents.tools import retrieve_hybrid_context, get_vietnam_stock_price
from app.llms.qwen_hf import QwenHFChat  # hoặc đường dẫn thực tế trong project của bạn
from app.llms.kaggle_api_chat import KaggleAPIChat
from langchain_core.messages import SystemMessage, HumanMessage

MAX_TOOL_CALLS = 3
MAX_ITERATIONS = 5

# llm = QwenHFChat(
#     model_name="Qwen/Qwen2.5-0.5B-Instruct",
#     temperature=0.0,
#     max_new_tokens=450,
# )
link_api = "https://lzbwj-34-68-22-16.run.pinggy-free.link/"
llm = KaggleAPIChat(
    api_url=link_api,
    api_key="my-secret-api-key-123",
    temperature=0.0,
    max_new_tokens=300,
)

# llm_with_tools = llm.bind_tools([retrieve_hybrid_context])
llm_with_tools = llm.bind_tools([
    retrieve_hybrid_context,
    get_vietnam_stock_price,
])


# =========================================================
# CONFIG
# =========================================================
REWRITE_MODE = "auto"  # "off" | "auto" | "always"
REWRITE_MAX_LEN = 120
REWRITE_MULTI_CLAUSE_MARKERS = [
    " và ", " hoặc ", " rồi ", " sau đó ", ",", ";", " bao gồm ", " gồm "
]

# =========================================================
# HELPERS VNSTOCK
# =========================================================
STOCK_QUERY_MARKERS = [
    "cổ phiếu",
    "co phieu",
    "chứng khoán",
    "chung khoan",
    "giá",
    "gia",
    "ohlcv",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "khối lượng",
    "khoi luong",
    "mã",
    "ma",
    "vnindex",
    "hnxindex",
    "upcomindex",
]

DATE_PATTERN = r"\b\d{4}-\d{2}-\d{2}\b"


def _looks_like_stock_query(text: str) -> bool:
    t = str(text or "").lower()

    has_stock_marker = any(marker in t for marker in STOCK_QUERY_MARKERS)
    has_date = re.search(DATE_PATTERN, t) is not None

    # Ví dụ: "giá FPT ngày 2024-05-24"
    # Có marker giá + có ngày thì gần như chắc là stock query.
    return has_stock_marker and has_date


def _extract_stock_args_rule_based(text: str) -> dict:
    """
    Extract đơn giản:
    - symbol: lấy mã viết hoa 2-10 ký tự.
    - date: lấy YYYY-MM-DD.
    """
    raw = str(text or "").strip()
    upper = raw.upper()

    date_match = re.search(DATE_PATTERN, raw)
    date = date_match.group(0) if date_match else ""

    # Ưu tiên các mã phổ biến dạng viết hoa.
    # Loại bớt các từ không phải mã.
    blacklist = {
        "NGAY", "NGÀY", "GIA", "GIÁ", "MA", "MÃ", "CHO", "LAY", "LẤY",
        "CO", "CỔ", "PHIEU", "PHIẾU", "CHUNG", "CHỨNG", "KHOAN", "KHOÁN",
        "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "OHLCV", "KBS", "VCI",
    }

    candidates = re.findall(r"\b[A-Z]{2,10}\b", upper)
    candidates = [c for c in candidates if c not in blacklist]

    symbol = candidates[0] if candidates else ""

    source = "KBS"
    if " VCI" in f" {upper} ":
        source = "VCI"
    elif " KBS" in f" {upper} ":
        source = "KBS"

    mode = "previous"
    if "đúng ngày" in raw.lower() or "dung ngay" in raw.lower() or "exact" in raw.lower():
        mode = "exact"
    elif "gần nhất" in raw.lower() or "gan nhat" in raw.lower() or "nearest" in raw.lower():
        mode = "nearest"

    return {
        "symbol": symbol,
        "date": date,
        "source": source,
        "mode": mode,
    }

# =========================================================
# HELPERS
# =========================================================
def _to_tool_call(parsed: dict):
    return {
        "name": parsed["tool_name"],
        "args": parsed.get("arguments", {}) or {},
        "id": str(uuid.uuid4()),
        "type": "tool_call",
    }


def _make_tool_call(name: str, args: dict):
    return {
        "name": name,
        "args": args or {},
        "id": str(uuid.uuid4()),
        "type": "tool_call",
    }


def _safe_extract_json_from_text(text: str):
    if not text:
        return None
    text = str(text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _normalize_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes"}:
            return True
        if v in {"false", "0", "no"}:
            return False
    return default


def _normalize_questions(questions, original_query: str):
    if isinstance(questions, str):
        questions = [questions]
    if not isinstance(questions, list):
        return [original_query]

    cleaned = []
    seen = set()
    for q in questions:
        q = str(q).strip()
        if not q:
            continue
        key = " ".join(q.lower().split())
        if key not in seen:
            seen.add(key)
            cleaned.append(q)

    return cleaned or [original_query]


def _normalize_clarification(value, default="Bạn có thể nói rõ câu hỏi hơn được không?"):
    if isinstance(value, str):
        v = value.strip()
        if v and v.lower() not in {"true", "false"}:
            return v
    return default


SUMMARY_INTENT_MARKERS = [
    "tóm tắt", "tổng quan", "tổng hợp", "khái quát",
    "summary", "overview", "kết luận",
    "nêu", "cho biết về", "mô tả về", "giới thiệu về"
]


def _should_use_rewrite(state: State) -> bool:
    if REWRITE_MODE == "always":
        return True
    if REWRITE_MODE == "off":
        return False

    last_message = state["messages"][-1]
    q = str(getattr(last_message, "content", "") or "").strip()
    if not q:
        return False

    q_lower = q.lower()
    conversation_summary = state.get("conversation_summary", "") or ""  # ← lấy từ state

    # Intent tóm tắt/liệt kê luôn rewrite
    if any(m in q_lower for m in SUMMARY_INTENT_MARKERS):
        return True
    if any(m in q_lower for m in ENUMERATION_MARKERS):  # ← thêm liệt kê
        return True

    marker_count = sum(1 for m in REWRITE_MULTI_CLAUSE_MARKERS if m in q_lower)

    if len(q) <= REWRITE_MAX_LEN and marker_count < 2:
        return False

    context_dependent_markers = [
        "ở trên", "bên trên", "đó", "cái đó", "phần đó",
        "tiếp theo", "còn", "thế", "vậy", "này", "kia"
    ]
    if conversation_summary and any(m in q_lower for m in context_dependent_markers):
        return True

    return len(q) > REWRITE_MAX_LEN or marker_count >= 2
    # ↑ Xóa hết code trùng lặp phía dưới, gộp vào 1 return


def _detect_intent(user_text: str) -> str:
    t = str(user_text or "").strip()
    tl = t.lower()

    if not tl:
        return "chitchat"

    if tl in {"hi", "hello", "hey", "xin chào", "chào", "chao", "alo", "yo"}:
        return "chitchat"

    if len(tl) <= 12 and any(x in tl for x in ["xin", "chào", "hello", "hi", "hey"]):
        return "chitchat"

    if any(k in tl for k in [
        "bạn là ai", "ban la ai", "tên bạn", "ten ban", "who are you", "your name",
        "how are you", "bạn khoẻ", "ban khoe", "chitchat", "trò chuyện", "tro chuyen"
    ]):
        return "chitchat"

    prompt = f"""
Bạn là bộ định tuyến intent cho chatbot RAG.

Nhiệm vụ:
Phân loại câu hỏi người dùng thành đúng 1 trong 2 nhãn:
1. "rag"
2. "chitchat"

Chỉ trả về JSON hợp lệ:
{{
  "type": "rag" | "chitchat",
  "reason": "ngắn gọn"
}}

User query:
"{t}"
"""

    try:
        raw = llm.with_config(temperature=0.0).invoke([
            SystemMessage(content="You are an intent router. Return valid JSON only."),
            HumanMessage(content=prompt)
        ])

        raw_text = getattr(raw, "content", raw)
        if isinstance(raw_text, list):
            parts = []
            for item in raw_text:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            raw_text = "\n".join(parts)

        parsed = _safe_extract_json_from_text(raw_text)
        if isinstance(parsed, dict):
            intent = str(parsed.get("type", "")).strip().lower()
            if intent in {"rag", "chitchat"}:
                return intent
    except Exception as e:
        print(f"⚠️ Intent routing error: {e}")

    return "rag"


# =========================================================
# MAIN PREP NODES
# =========================================================
def summarize_history(state: State):
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}

    relevant_msgs = [
        msg for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
    ]

    if not relevant_msgs:
        return {"conversation_summary": ""}

    conversation = "Conversation history:\n"
    for msg in relevant_msgs[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"

    summary_response = llm.with_config(temperature=0.2).invoke([
        SystemMessage(content=get_conversation_summary_prompt()),
        HumanMessage(content=conversation)
    ])
    return {
        "conversation_summary": summary_response.content,
        "agent_answers": [{"__reset__": True}]
    }


def rewrite_query(state: State):
    last_message = state["messages"][-1]
    q = str(getattr(last_message, "content", "") or "").strip()
    conversation_summary = state.get("conversation_summary", "") or ""

    intent = _detect_intent(q)

    if intent == "chitchat":
        return {
            "intent": intent,
            "questionIsClear": True,
            "originalQuery": q or "Xin chào!",
            "rewrittenQuestions": [q or "Xin chào!"]
        }

    # Fast path: không rewrite nếu không cần
    if not _should_use_rewrite(state):
        delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
        return {
            "intent": intent,
            "questionIsClear": True,
            "messages": delete_all,
            "originalQuery": q,
            "rewrittenQuestions": [q]
        }

    context_section = ""
    if str(conversation_summary).strip():
        context_section += f"Conversation Context:\n{conversation_summary.strip()}\n\n"
    context_section += f"User Query:\n{q}\n"

    # Structured output path
    try:
        llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(QueryAnalysis)
        response = llm_with_structure.invoke([
            SystemMessage(content=get_rewrite_query_prompt()),
            HumanMessage(content=context_section)
        ])

        questions = _normalize_questions(getattr(response, "questions", None), q)
        is_clear = _normalize_bool(getattr(response, "is_clear", True), default=True)
        clarification = _normalize_clarification(
            getattr(response, "clarification_needed", None),
            default="Bạn có thể nói rõ câu hỏi hơn được không?"
        )

        if is_clear:
            delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
            return {
                "intent": intent,
                "questionIsClear": True,
                "messages": delete_all,
                "originalQuery": q,
                "rewrittenQuestions": questions
            }

        return {
            "intent": intent,
            "questionIsClear": False,
            "messages": [AIMessage(content=clarification)],
            "originalQuery": q,
            "rewrittenQuestions": [q]
        }

    except Exception:
        pass

    # Raw JSON fallback
    try:
        raw = llm.with_config(temperature=0.1).invoke([
            SystemMessage(
                content=get_rewrite_query_prompt() + "\nReturn valid JSON only with keys: is_clear, questions, clarification_needed."),
            HumanMessage(content=context_section)
        ])
        raw_text = getattr(raw, "content", raw)
        parsed = _safe_extract_json_from_text(raw_text)

        if isinstance(parsed, dict):
            questions = _normalize_questions(parsed.get("questions"), q)
            is_clear = _normalize_bool(parsed.get("is_clear", True), default=True)
            clarification = _normalize_clarification(
                parsed.get("clarification_needed"),
                default="Bạn có thể nói rõ câu hỏi hơn được không?"
            )

            if is_clear:
                delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
                return {
                    "intent": intent,
                    "questionIsClear": True,
                    "messages": delete_all,
                    "originalQuery": q,
                    "rewrittenQuestions": questions
                }

            return {
                "intent": intent,
                "questionIsClear": False,
                "messages": [AIMessage(content=clarification)],
                "originalQuery": q,
                "rewrittenQuestions": [q]
            }
    except Exception:
        pass

    # Fallback cuối: coi như rõ, không được trả "True"
    delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
    return {
        "intent": intent,
        "questionIsClear": True,
        "messages": delete_all,
        "originalQuery": q,
        "rewrittenQuestions": [q]
    }


# def request_clarification(state: State):
#     return {}

def request_clarification(state: State):
    clarification = "Bạn có thể nói rõ hơn bạn muốn hỏi phần nào trong tài liệu không?"
    return {"messages": [AIMessage(content=clarification)]}


# =========================================================
# AGENT NODES
# =========================================================
# def orchestrator(state: AgentState):
#     question = state["question"]
#     context_summary = state.get("context_summary", "").strip()
#
#     if not state.get("messages"):
#         # Detect và enrich query ngay tại đây, chỉ 1 lần
#         q_lower = question.lower()
#
#         is_summary = any(m in q_lower for m in [
#             "tóm tắt", "tổng quan", "kết luận", "summary", "overview", "phát hiện chính"
#         ])
#         is_enum = any(m in q_lower for m in [
#             "gồm gì", "gồm những", "bao gồm gì", "liệt kê", "những gì", "gồm có", "nêu ra"
#         ])
#
#         retrieval_query = question
#         extra_parents = 3
#         if is_summary:
#             retrieval_query = f"{question} kết luận phần tóm tắt"
#             extra_parents = 5
#         elif is_enum:
#             extra_parents = 4
#
#         human_msg = HumanMessage(content=question)
#         forced_tool_call = _make_tool_call("retrieve_hybrid_context", {
#             "query": retrieval_query,
#             "max_parents": extra_parents,
#         })
#         ai_msg = AIMessage(content="", tool_calls=[forced_tool_call])
#
#         return {
#             "messages": [human_msg, ai_msg],
#             "tool_call_count": 1,
#             "iteration_count": state.get("iteration_count", 0) + 1,
#         }
#
#     # Second turn trở đi — để LLM quyết định
#     sys_msg = SystemMessage(content=get_orchestrator_prompt())
#     summary_injection = (
#         [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
#         if context_summary else []
#     )
#     raw = llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"])
#     tool_calls = getattr(raw, "tool_calls", None) or []
#     if not tool_calls:
#         parsed = _safe_extract_json_from_text(getattr(raw, "content", "") or "")
#         if isinstance(parsed, dict) and parsed.get("tool_name"):
#             tool_calls = [_to_tool_call(parsed)]
#
#     ai_msg = AIMessage(
#         content=getattr(raw, "content", "") or "",
#         tool_calls=tool_calls
#     )
#     return {
#         "messages": [ai_msg],
#         "tool_call_count": len(tool_calls),
#         "iteration_count": state.get("iteration_count", 0) + 1,
#     }

def orchestrator(state: AgentState):
    question = state["question"]
    context_summary = state.get("context_summary", "").strip()

    if not state.get("messages"):
        # ── Detect intent một lần duy nhất ──────────────────────────────
        q_lower = question.lower()

        # ================================================================
        # BRANCH 1: Stock query → gọi vnstock tool
        # ================================================================
        if _looks_like_stock_query(question):
            stock_args = _extract_stock_args_rule_based(question)
            if stock_args.get("symbol") and stock_args.get("date"):
                # Có đủ args → force gọi vnstock
                human_msg = HumanMessage(content=question)
                forced_tool_call = _make_tool_call(
                    "get_vietnam_stock_price",
                    stock_args,
                )
                ai_msg = AIMessage(content="", tool_calls=[forced_tool_call])
                return {
                    "messages": [human_msg, ai_msg],
                    "tool_call_count": 1,
                    "iteration_count": state.get("iteration_count", 0) + 1,
                }
            # Thiếu symbol/date → không force, để LLM xử lý ở second turn
            # bằng cách KHÔNG return sớm, tiếp tục xuống RAG bên dưới

        # ================================================================
        # BRANCH 2: RAG query → giữ nguyên logic cũ (tốt)
        # ================================================================
        is_summary = any(m in q_lower for m in [
            "tóm tắt", "tổng quan", "kết luận", "summary", "overview", "phát hiện chính",
        ])
        is_enum = any(m in q_lower for m in [
            "gồm gì", "gồm những", "bao gồm gì", "liệt kê",
            "những gì", "gồm có", "nêu ra",
        ])

        retrieval_query = question
        extra_parents = 3
        if is_summary:
            retrieval_query = f"{question} kết luận phần tóm tắt"
            extra_parents = 5
        elif is_enum:
            extra_parents = 4

        human_msg = HumanMessage(content=question)
        forced_tool_call = _make_tool_call(
            "retrieve_hybrid_context",
            {
                "query": retrieval_query,
                "user_id": state.get("user_id", "anonymous"),
                "role": state.get("role", "employee"),
                "max_parents": extra_parents,
            },
        )
        ai_msg = AIMessage(content="", tool_calls=[forced_tool_call])
        return {
            "messages": [human_msg, ai_msg],
            "tool_call_count": 1,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    # ====================================================================
    # Second turn trở đi → để LLM quyết định (giữ nguyên code cũ)
    # ====================================================================
    sys_msg = SystemMessage(content=get_orchestrator_prompt())
    summary_injection = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
        if context_summary else []
    )
    raw = llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"])
    tool_calls = getattr(raw, "tool_calls", None) or []
    for tool_call in tool_calls:
        if tool_call.get("name") == "retrieve_hybrid_context":
            tool_call.setdefault("args", {})
            tool_call["args"]["user_id"] = state.get("user_id", "anonymous")
            tool_call["args"]["role"] = state.get("role", "employee")

    if not tool_calls:
        parsed = _safe_extract_json_from_text(getattr(raw, "content", "") or "")
        if isinstance(parsed, dict) and parsed.get("tool_name"):
            tool_calls = [_to_tool_call(parsed)]

    ai_msg = AIMessage(
        content=getattr(raw, "content", "") or "",
        tool_calls=tool_calls,
    )
    return {
        "messages": [ai_msg],
        "tool_call_count": len(tool_calls),
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def route_after_orchestrator_call(
        state: AgentState
) -> Literal["tools", "fallback_response", "collect_answer"]:
    iteration = state.get("iteration_count", 0)
    tool_count = state.get("tool_call_count", 0)

    if iteration >= MAX_ITERATIONS or tool_count > MAX_TOOL_CALLS:
        return "fallback_response"

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        return "collect_answer"

    return "tools"


ENUMERATION_MARKERS = [
    "gồm gì", "gồm những gì", "bao gồm gì", "bao gồm những gì",
    "có gì", "những gì", "gì gì", "liệt kê", "gồm có",
    "kể ra", "nêu ra", "nêu bật", "cho biết gồm"
]
SUMMARY_MARKERS = [
    "tóm tắt", "tổng quan", "kết luận", "phát hiện chính", "summary", "overview"
]


# def answer_from_context(state: AgentState):
#     latest_payload = None
#     for msg in reversed(state["messages"]):
#         if isinstance(msg, ToolMessage):
#             try:
#                 data = json.loads(msg.content)
#                 if isinstance(data, dict) and "final_context" in data:
#                     latest_payload = data
#                     break
#             except Exception:
#                 pass
#
#     if not latest_payload:
#         return {"messages": [AIMessage(content="Không tìm thấy ngữ cảnh phù hợp.")]}
#
#     final_context = latest_payload.get("final_context", "")
#     question = state.get("question", "")
#     q_lower = question.lower()
#
#     is_enum = any(m in q_lower for m in ENUMERATION_MARKERS)
#     is_summary = any(m in q_lower for m in SUMMARY_MARKERS)
#
#     if is_enum:
#         answer_instruction = (
#             "Câu hỏi yêu cầu LIỆT KÊ. Hãy trình bày ĐẦY ĐỦ từng mục từ ngữ cảnh. "
#             "KHÔNG bỏ sót bất kỳ mục nào dù nhỏ. KHÔNG gộp hay tóm gọn. "
#             "Giữ nguyên mọi con số, tỷ lệ phần trăm, mốc thời gian. "
#             "Trình bày dạng danh sách gạch đầu dòng."
#         )
#         max_tokens = 800
#     elif is_summary:
#         answer_instruction = (
#             "Câu hỏi yêu cầu TỔNG HỢP. Trình bày đầy đủ các phát hiện và kết luận. "
#             "Không rút gọn bất kỳ điểm nào."
#         )
#         max_tokens = 700
#     else:
#         answer_instruction = (
#             "Trả lời trực tiếp và đúng trọng tâm. "
#             "Giữ nguyên mọi số liệu và chi tiết quan trọng."
#         )
#         max_tokens = 450
#
#     # Tăng token động theo intent
#     llm._model.generation_config.max_new_tokens = max_tokens
#
#     resp = llm.invoke([
#         SystemMessage(content=(
#             "Bạn là trợ lý RAG. BẮT BUỘC trả lời bằng tiếng Việt. "
#             "KHÔNG được bỏ sót thông tin khi câu hỏi yêu cầu liệt kê hoặc tổng hợp."
#         )),
#         HumanMessage(content=(
#             f"Câu hỏi: {question}\n\n"
#             f"Ngữ cảnh:\n{final_context}\n\n"
#             f"{answer_instruction}"
#         ))
#     ])
#
#     # Reset về default
#     llm._model.generation_config.max_new_tokens = 450
#
#     answer = getattr(resp, "content", str(resp)).strip()
#     return {"messages": [AIMessage(content=answer or "Tài liệu không nêu rõ.")]}


def answer_from_context(state: AgentState):
    latest_payload = None

    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                if isinstance(data, dict):
                    latest_payload = data
                    break
            except Exception:
                pass

    if not latest_payload:
        return {"messages": [AIMessage(content="Không tìm thấy dữ liệu phù hợp.")]}

    question = state.get("question", "")

    # =====================================================
    # 1) Trả lời từ stock tool
    # =====================================================
    if latest_payload.get("type") == "stock_price":
        if not latest_payload.get("ok", False):
            error = latest_payload.get("error", "Không lấy được dữ liệu cổ phiếu.")
            return {"messages": [AIMessage(content=error)]}

        resp = llm.invoke([
            SystemMessage(content=(
                "Bạn là trợ lý phân tích dữ liệu chứng khoán Việt Nam. "
                "Hãy trả lời bằng tiếng Việt, rõ ràng, ngắn gọn. "
                "Chỉ sử dụng dữ liệu được cung cấp, không tự suy đoán, "
                "không đưa ra khuyến nghị mua bán."
            )),
            HumanMessage(content=(
                f"Câu hỏi người dùng:\n{question}\n\n"
                f"Dữ liệu OHLCV lấy được:\n"
                f"{json.dumps(latest_payload, ensure_ascii=False, indent=2)}\n\n"
                "Hãy trình bày kết quả gồm: mã, ngày yêu cầu, ngày giao dịch thực tế, "
                "open, high, low, close, volume. "
                "Nếu ngày giao dịch thực tế khác ngày yêu cầu, hãy nói rõ lý do có thể là "
                "ngày nghỉ/cuối tuần/không có phiên giao dịch."
            ))
        ])

        answer = getattr(resp, "content", str(resp)).strip()
        return {"messages": [AIMessage(content=answer)]}

    # =====================================================
    # 2) Trả lời từ RAG tool như logic cũ
    # =====================================================
    if "final_context" not in latest_payload:
        return {"messages": [AIMessage(content="Không tìm thấy ngữ cảnh phù hợp.")]}

    final_context = latest_payload.get("final_context", "")
    q_lower = question.lower()

    is_enum = any(m in q_lower for m in ENUMERATION_MARKERS)
    is_summary = any(m in q_lower for m in SUMMARY_MARKERS)

    if is_enum:
        answer_instruction = (
            "Câu hỏi yêu cầu LIỆT KÊ. "
            "Hãy trình bày ĐẦY ĐỦ từng mục từ ngữ cảnh. "
            "KHÔNG bỏ sót bất kỳ mục nào dù nhỏ. KHÔNG gộp hay tóm gọn. "
            "Giữ nguyên mọi con số, tỷ lệ phần trăm, mốc thời gian. "
            "Trình bày dạng danh sách gạch đầu dòng."
        )
        max_tokens = 800
    elif is_summary:
        answer_instruction = (
            "Câu hỏi yêu cầu TỔNG HỢP. Trình bày đầy đủ các phát hiện và kết luận. "
            "Không rút gọn bất kỳ điểm nào."
        )
        max_tokens = 700
    else:
        answer_instruction = (
            "Trả lời trực tiếp và đúng trọng tâm. "
            "Giữ nguyên mọi số liệu và chi tiết quan trọng."
        )
        max_tokens = 450

    llm._model.generation_config.max_new_tokens = max_tokens

    resp = llm.invoke([
        SystemMessage(content=(
            "Bạn là trợ lý RAG. BẮT BUỘC trả lời bằng tiếng Việt. "
            "KHÔNG được bỏ sót thông tin khi câu hỏi yêu cầu liệt kê hoặc tổng hợp."
        )),
        HumanMessage(content=(
            f"Câu hỏi: {question}\n\n"
            f"Ngữ cảnh:\n{final_context}\n\n"
            f"{answer_instruction}"
        ))
    ])

    llm._model.generation_config.max_new_tokens = 450

    answer = getattr(resp, "content", str(resp)).strip()
    return {"messages": [AIMessage(content=answer or "Tài liệu không nêu rõ.")]}

def fallback_response(state: AgentState):
    seen = set()
    unique_contents = []

    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_contents.append(m.content)
            seen.add(m.content)

    context_text = (
        "\n\n".join(f"--- NGUỒN DỮ LIỆU {i} ---\n{content}" for i, content in enumerate(unique_contents, 1))
        if unique_contents else
        "Không có dữ liệu nào được truy xuất từ tài liệu."
    )

    prompt_content = (
        f"Câu hỏi của người dùng:\n{state.get('question')}\n\n"
        f"{context_text}\n\n"
        "Yêu cầu:\n"
        "- Chỉ sử dụng dữ liệu ở trên\n"
        "- Trả lời hoàn toàn bằng tiếng Việt\n"
        "- Không tự suy diễn thêm nếu tài liệu không nêu rõ\n"
        "- Nếu thiếu dữ liệu, hãy nói rõ bằng tiếng Việt rằng tài liệu không cung cấp đủ thông tin"
    )

    response = llm.invoke([
        SystemMessage(content=(
            "Bạn là trợ lý RAG.\n"
            "BẮT BUỘC trả lời hoàn toàn bằng tiếng Việt.\n"
            "Không dùng tiếng Anh để diễn giải."
        )),
        HumanMessage(content=prompt_content)
    ])

    answer = getattr(response, "content", str(response)).strip()
    return {"messages": [AIMessage(content=answer or "Tài liệu không nêu rõ.")]}


def collect_answer(state: AgentState):
    last_message = state["messages"][-1]

    is_valid = (
            isinstance(last_message, AIMessage)
            and bool(last_message.content)
            and not getattr(last_message, "tool_calls", None)
    )

    answer = last_message.content if is_valid else "Không thể tạo câu trả lời."

    return {
        "final_answer": answer,
        "agent_answers": [
            {
                "index": state["question_index"],
                "question": state["question"],
                "answer": answer
            }
        ]
    }


# =========================================================
# MAIN GRAPH NODES
# =========================================================
def chitchat(state: State):
    last = state["messages"][-1]
    sys = SystemMessage(content=
                        "Bạn là một trợ lý thân thiện và hữu ích. "
                        "Hãy trả lời một cách tự nhiên, giống hội thoại, và ngắn gọn. "
                        "Nếu người dùng chào bạn, hãy chào lại. "
                        "Nếu người dùng hỏi bạn là ai, hãy giới thiệu ngắn gọn về bản thân."
                        )
    resp = llm.with_config(temperature=0.6).invoke([sys, last])
    return {"messages": [AIMessage(content=getattr(resp, "content", str(resp)))]}


def route_after_rewrite(state: State):
    if not state.get("questionIsClear", False):
        return "request_clarification"

    if state.get("intent", "rag") == "chitchat":
        return "chitchat"

    questions = state.get("rewrittenQuestions", []) or []
    questions = [q.strip() for q in questions if str(q).strip()]

    deduped = []
    seen = set()
    for q in questions:
        key = " ".join(q.lower().split())
        if key not in seen:
            seen.add(key)
            deduped.append(q)

    # deduped = deduped[:1]

    # SAU:
    deduped = deduped[:3]

    # return [
    #     Send("agent", {"question": query, "question_index": idx, "messages": []})
    #     for idx, query in enumerate(deduped)
    # ]
    return [
        Send(
            "agent",
            {
                "question": query,
                "question_index": idx,
                "messages": [],
                "user_id": state.get("user_id", "anonymous"),
                "role": state.get("role", "employee"),
            },
        )
        for idx, query in enumerate(deduped)
    ]

def aggregate_answers(state: State):
    answers = state.get("agent_answers", []) or []
    answers = [a for a in answers if isinstance(a, dict) and "answer" in a]

    if not answers:
        msg = (
            "Mình không tìm thấy thông tin liên quan trong tài liệu đã cung cấp. "
            "Bạn có thể nói rõ bạn muốn tìm trong tài liệu nào hoặc phần nào."
        )
        return {"messages": [AIMessage(content=msg)]}

    answers_sorted = sorted(answers, key=lambda x: x.get("index", 0))

    if len(answers_sorted) == 1:
        final = answers_sorted[0].get("answer", "")
        return {"messages": [AIMessage(content=final)]}

    stitched = []
    for a in answers_sorted:
        q = a.get("question", "")
        ans = a.get("answer", "")
        stitched.append(f"**{q}**\n{ans}")

    final = "\n\n---\n\n".join(stitched)
    return {"messages": [AIMessage(content=final)]}
