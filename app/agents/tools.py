import json
from langchain_core.tools import tool
import uuid


# =========================================================
# RETRIEVAL SETUP
# =========================================================
from app.ingestion.indexing import (
    build_child_vector_store,
    add_pdf_to_db,
    QDRANT_PATH,
    CHILD_COLLECTION,
)
from app.ingestion.pdf_to_markdown import PARENT_STORE_PATH
from app.retrieval.context_builder import (
    retrieve_parent_docs_by_hybrid,
    build_main_secondary_context,
)


def build_retriever(child_vector_store, k: int = 20):
    """Wrap vector store thành retriever có .invoke()"""
    return child_vector_store.as_retriever(search_kwargs={"k": k})


# =========================================================
# KHỞI TẠO RETRIEVER TOÀN CỤC (dùng trong tools.py)
# =========================================================
def get_child_hybrid_retriever(k: int = 20):
    client, child_vector_store = build_child_vector_store(
        collection_name=CHILD_COLLECTION,
        qdrant_path=QDRANT_PATH,
        recreate=False,
    )
    retriever = build_retriever(child_vector_store, k=k)
    return client, retriever


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

@tool
def retrieve_hybrid_context(
        query: str,
        max_parents: int = 3,
        rank_constant: int = 20,
        use_secondary_context: bool = True,
        max_secondary_parents: int = 1,
        secondary_ratio_threshold: float = 0.55,
) -> str:
    """Retrieve relevant document context using hybrid search for a given query."""
    q_lower = query.lower()

    is_summary = any(m in q_lower for m in ["tóm tắt", "tổng quan", "kết luận"])
    is_enum = any(m in q_lower for m in [
        "gồm gì", "gồm những", "bao gồm gì", "liệt kê", "những gì", "nêu ra", "gồm có"
    ])

    if is_summary:
        max_parents = max(max_parents, 5)
        max_secondary_parents = 3
        secondary_ratio_threshold = 0.25
    elif is_enum:  # ← THÊM khối này
        max_parents = max(max_parents, 4)
        max_secondary_parents = 2
        secondary_ratio_threshold = 0.35

    client, child_hybrid_retriever = get_child_hybrid_retriever(k=20)

    try:
        results = retrieve_parent_docs_by_hybrid(
            query=query,
            child_retriever=child_hybrid_retriever,
            parent_store_path=PARENT_STORE_PATH,
            max_parents=max_parents,
            rank_constant=rank_constant,
        )
    finally:
        client.close()

    context_data = build_main_secondary_context(
        query=query,
        results=results,
        use_secondary_context=use_secondary_context,
        max_secondary_parents=max_secondary_parents,
        secondary_ratio_threshold=secondary_ratio_threshold,
    )

    # Optional: keep only lightweight metadata to avoid huge tool payload
    compact_results = []
    for item in results:
        compact_results.append({
            "parent_id": item.get("parent_id"),
            "score": item.get("score"),
            "child_preview": str(item.get("best_child_text", ""))[:500]
        })

    payload = {
        "query": query,
        "main_context": context_data.get("main_context", ""),
        "secondary_context": context_data.get("secondary_context", ""),
        "final_context": context_data.get("final_context", ""),
        "results_meta": compact_results,
    }

    return json.dumps(payload, ensure_ascii=False)
