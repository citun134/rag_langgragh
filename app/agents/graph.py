import time
from functools import wraps
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from .nodes import (summarize_history, rewrite_query, request_clarification,
                    aggregate_answers, chitchat, orchestrator, answer_from_context,
                    fallback_response, collect_answer, route_after_orchestrator_call,
                    route_after_rewrite,
                    )
from .state import State, AgentState
from .tools import retrieve_hybrid_context, get_vietnam_stock_price

# =========================================================
# WRAP NODES FOR TIMING
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


summarize_history = timed("summarize_history")(summarize_history)
rewrite_query = timed("rewrite_query")(rewrite_query)
request_clarification = timed("request_clarification")(request_clarification)
aggregate_answers = timed("aggregate_answers")(aggregate_answers)
chitchat = timed("chitchat")(chitchat)

orchestrator = timed("orchestrator")(orchestrator)
answer_from_context = timed("answer_from_context")(answer_from_context)
fallback_response = timed("fallback_response")(fallback_response)
collect_answer = timed("collect_answer")(collect_answer)

# =========================================================
# BUILD + COMPILE GRAPH
# =========================================================
checkpointer = InMemorySaver()

agent_builder = StateGraph(AgentState)
agent_builder.add_node("orchestrator", orchestrator)
# agent_builder.add_node("tools", ToolNode([retrieve_hybrid_context]))
agent_builder.add_node(
    "tools",
    ToolNode([
        retrieve_hybrid_context,
        get_vietnam_stock_price,
    ])
)
agent_builder.add_node("answer_from_context", answer_from_context)
agent_builder.add_node("fallback_response", fallback_response)
agent_builder.add_node("collect_answer", collect_answer)

agent_builder.add_edge(START, "orchestrator")
agent_builder.add_conditional_edges(
    "orchestrator",
    route_after_orchestrator_call,
    {
        "tools": "tools",
        "fallback_response": "fallback_response",
        "collect_answer": "collect_answer",
    },
)
agent_builder.add_edge("tools", "answer_from_context")
agent_builder.add_edge("answer_from_context", "collect_answer")
agent_builder.add_edge("fallback_response", "collect_answer")
agent_builder.add_edge("collect_answer", END)
agent_subgraph = agent_builder.compile()

graph_builder = StateGraph(State)
graph_builder.add_node("summarize_history", summarize_history)
graph_builder.add_node("rewrite_query", rewrite_query)
graph_builder.add_node("request_clarification", request_clarification)
graph_builder.add_node("chitchat", chitchat)
graph_builder.add_node("agent", agent_subgraph)
graph_builder.add_node("aggregate_answers", aggregate_answers)

graph_builder.add_edge(START, "summarize_history")
graph_builder.add_edge("summarize_history", "rewrite_query")
graph_builder.add_conditional_edges(
    "rewrite_query",
    route_after_rewrite,
    {
        "request_clarification": "request_clarification",
        "agent": "agent",
        "chitchat": "chitchat",
    },
)
graph_builder.add_edge("request_clarification", "rewrite_query")
graph_builder.add_edge("agent", "aggregate_answers")
graph_builder.add_edge("aggregate_answers", END)
graph_builder.add_edge("chitchat", END)

agent_graph = graph_builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["request_clarification"]
)

# display(Image(agent_graph.get_graph(xray=False).draw_mermaid_png()))
print("✓ Agent graph compiled successfully.")