import uuid
import json
import argparse

from langchain_core.messages import HumanMessage, AIMessage


# =========================================================
# ROLE CONFIG
# =========================================================
ROLE_ALLOWED_VISIBILITIES = {
    "employee": ["public"],
    "manager": ["public", "internal"],
    "hr": ["public", "internal", "confidential"],
    "admin": ["public", "internal", "confidential", "private"],
}


def normalize_role(role: str) -> str:
    """
    Normalize user role.

    Supported roles:
    - employee
    - manager
    - hr
    - admin
    """

    role = str(role or "employee").lower().strip()

    if role not in ROLE_ALLOWED_VISIBILITIES:
        print(f"⚠️ Unknown role '{role}', fallback to 'employee'.")
        return "employee"

    return role


def get_final_answer(result: dict) -> str:
    """
    Extract final assistant answer from LangGraph result.
    """

    messages = result.get("messages", [])

    for message in reversed(messages):
        if (
            isinstance(message, AIMessage)
            and message.content
            and not getattr(message, "tool_calls", None)
        ):
            return message.content.strip()

    return "Không tìm thấy câu trả lời."


def print_debug_messages(result: dict) -> None:
    """
    Print all messages for debugging LangGraph flow.
    """

    print("\n=== FINAL STATE KEYS ===")
    print(list(result.keys()))

    print("\n=== ALL MESSAGES DEBUG ===")

    messages = result.get("messages", [])

    for index, message in enumerate(messages, 1):
        role = getattr(message, "type", message.__class__.__name__)
        content = getattr(message, "content", str(message))
        tool_calls = getattr(message, "tool_calls", None)

        print(f"\n[{index}] {role}:")

        if tool_calls:
            print(
                "tool_calls:",
                json.dumps(
                    tool_calls,
                    ensure_ascii=False,
                    indent=2,
                )[:1000],
            )

        print(str(content)[:2000])


# =========================================================
# AGENT GRAPH RUN
# =========================================================
def run_agent(
    user_query: str,
    user_id: str = "anonymous",
    role: str = "employee",
    thread_id: str | None = None,
    recursion_limit: int = 50,
    debug_messages: bool = False,
) -> str:
    """
    Run agent_graph with access-control information.

    Access rules are handled later in retriever/tool layer:
    - employee: public + own private
    - manager: public + internal + own private
    - hr: public + internal + confidential + own private
    - admin: all documents

    Args:
        user_query: User question.
        user_id: Current user id, for private document ownership.
        role: Current user role.
        thread_id: LangGraph thread id.
        recursion_limit: LangGraph recursion limit.
        debug_messages: Whether to print full message trace.

    Returns:
        Final answer string.
    """

    from app.agents.graph import agent_graph  # noqa: E402

    role = normalize_role(role)

    config = {
        "configurable": {
            "thread_id": thread_id or str(uuid.uuid4()),
        },
        "recursion_limit": recursion_limit,
    }

    graph_input = {
        "question": user_query,
        "messages": [
            HumanMessage(content=user_query),
        ],

        # NEW ACCESS CONTROL FIELDS
        "user_id": user_id,
        "role": role,
    }

    result = agent_graph.invoke(
        graph_input,
        config=config,
    )

    if debug_messages:
        print_debug_messages(result)

    return get_final_answer(result)


# =========================================================
# MAIN — TEST FULL PIPELINE
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run RAG LangGraph agent with role-based access control."
    )

    parser.add_argument(
        "--query",
        type=str,
        default="xoài là gì",
        help="User question.",
    )

    parser.add_argument(
        "--user-id",
        type=str,
        default="alice",
        help="Current user id.",
    )

    parser.add_argument(
        "--role",
        type=str,
        default="employee",
        choices=["employee", "manager", "hr", "admin"],
        help="Current user role.",
    )

    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help="Optional LangGraph thread id.",
    )

    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=50,
        help="LangGraph recursion limit.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print all LangGraph messages.",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("AGENT GRAPH — FULL PIPELINE WITH ACCESS CONTROL")
    print("=" * 60)

    print(f"User ID : {args.user_id}")
    print(f"Role    : {args.role}")
    print(f"Allowed : {ROLE_ALLOWED_VISIBILITIES[normalize_role(args.role)]}")
    print(f"Query   : {args.query}")

    final_answer = run_agent(
        user_query=args.query,
        user_id=args.user_id,
        role=args.role,
        thread_id=args.thread_id,
        recursion_limit=args.recursion_limit,
        debug_messages=args.debug,
    )

    print("\n=== FINAL ANSWER ===")
    print(final_answer)


if __name__ == "__main__":
    main()