import time
from functools import wraps
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

from .nodes import (
    summarize_history,
    rewrite_query,
    request_clarification,
    aggregate_answers,
    chitchat,
    orchestrator,
    grade_context,
    answer_from_context,
    no_context_response,
    fallback_response,
    collect_answer,
    route_after_orchestrator_call,
    route_after_context_grade,
    route_after_rewrite,
)
from .state import State, AgentState
from .tools import retrieve_hybrid_context, get_vietnam_stock_price


# =========================================================
# TIMING WRAPPER
# =========================================================
def timed(name=None):
    def deco(fn):
        label = name or fn.__name__

        @wraps(fn)
        def wrap(state):
            t0 = time.time()
            out = fn(state)
            dt = time.time() - t0
            print(f"[NODE] {label}: {dt:.2f}s")
            return out

        return wrap

    return deco


def _timed_nodes():
    return {
        "summarize_history": timed("summarize_history")(summarize_history),
        "rewrite_query": timed("rewrite_query")(rewrite_query),
        "request_clarification": timed("request_clarification")(request_clarification),
        "chitchat": timed("chitchat")(chitchat),
        "aggregate_answers": timed("aggregate_answers")(aggregate_answers),
        "orchestrator": timed("orchestrator")(orchestrator),
        "grade_context": timed("grade_context")(grade_context),
        "answer_from_context": timed("answer_from_context")(answer_from_context),
        "no_context_response": timed("no_context_response")(no_context_response),
        "fallback_response": timed("fallback_response")(fallback_response),
        "collect_answer": timed("collect_answer")(collect_answer),
    }


NODES = _timed_nodes()

# =========================================================
# CHECKPOINTER
# =========================================================
# Demo/local: InMemorySaver. Production nên thay bằng SQLite/Postgres/Redis checkpointer.
checkpointer = InMemorySaver()


# =========================================================
# AGENT SUBGRAPH: xử lý một rewritten question
# =========================================================
def build_agent_subgraph():
    builder = StateGraph(AgentState)

    builder.add_node("orchestrator", NODES["orchestrator"])
    builder.add_node("tools", ToolNode([
        retrieve_hybrid_context,
        get_vietnam_stock_price,
    ]))
    builder.add_node("grade_context", NODES["grade_context"])
    builder.add_node("answer_from_context", NODES["answer_from_context"])
    builder.add_node("no_context_response", NODES["no_context_response"])
    builder.add_node("fallback_response", NODES["fallback_response"])
    builder.add_node("collect_answer", NODES["collect_answer"])

    builder.add_edge(START, "orchestrator")

    builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator_call,
        {
            "tools": "tools",
            "fallback_response": "fallback_response",
            "collect_answer": "collect_answer",
        },
    )

    # Điểm cải thiện chính:
    # Tool không đi thẳng vào LLM nữa, mà qua quality gate trước.
    builder.add_edge("tools", "grade_context")
    builder.add_conditional_edges(
        "grade_context",
        route_after_context_grade,
        {
            "answer_from_context": "answer_from_context",
            "no_context_response": "no_context_response",
        },
    )

    builder.add_edge("answer_from_context", "collect_answer")
    builder.add_edge("no_context_response", "collect_answer")
    builder.add_edge("fallback_response", "collect_answer")
    builder.add_edge("collect_answer", END)

    return builder.compile()


agent_subgraph = build_agent_subgraph()


# =========================================================
# MAIN GRAPH: xử lý hội thoại tổng thể
# =========================================================
def build_agent_graph():
    builder = StateGraph(State)

    builder.add_node("summarize_history", NODES["summarize_history"])
    builder.add_node("rewrite_query", NODES["rewrite_query"])
    builder.add_node("request_clarification", NODES["request_clarification"])
    builder.add_node("chitchat", NODES["chitchat"])
    builder.add_node("agent", agent_subgraph)
    builder.add_node("aggregate_answers", NODES["aggregate_answers"])

    builder.add_edge(START, "summarize_history")
    builder.add_edge("summarize_history", "rewrite_query")

    builder.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        {
            "request_clarification": "request_clarification",
            "agent": "agent",
            "chitchat": "chitchat",
        },
    )

    # Clarification là final response cho request hiện tại.
    # Không quay lại rewrite_query để tránh loop khi API không có resume/interrupt.
    builder.add_edge("request_clarification", END)
    builder.add_edge("chitchat", END)
    builder.add_edge("agent", "aggregate_answers")
    builder.add_edge("aggregate_answers", END)

    return builder.compile(checkpointer=checkpointer)


agent_graph = build_agent_graph()

print("✓ Optimized agent graph compiled successfully.")
