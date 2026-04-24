from pathlib import Path
from app.ingestion.indexing import build_child_vector_store, add_pdf_to_db, QDRANT_PATH, CHILD_COLLECTION
from app.ingestion.pdf_to_markdown import PARENT_STORE_PATH
from app.retrieval.context_builder import retrieve_parent_docs_by_hybrid, build_main_secondary_context

def build_retriever(child_vector_store, k: int = 20):
    """Wrap vector store thành retriever có .invoke()"""
    return child_vector_store.as_retriever(search_kwargs={"k": k})


def query_with_context(
    query: str,
    child_retriever,
    parent_store_path: str | Path = PARENT_STORE_PATH,
    max_parents: int = 3,
    use_secondary_context: bool = True,
    max_secondary_parents: int = 1,
    secondary_ratio_threshold: float = 0.55,
):
    results = retrieve_parent_docs_by_hybrid(
        query=query,
        child_retriever=child_retriever,
        parent_store_path=str(parent_store_path),
        max_parents=max_parents,
        rank_constant=20,
    )

    context_data = build_main_secondary_context(
        query=query,
        results=results,
        use_secondary_context=use_secondary_context,
        max_secondary_parents=max_secondary_parents,
        secondary_ratio_threshold=secondary_ratio_threshold,
    )

    return context_data


def main():

    # pdf_path = r"C:\RAG_PRO\data\xoai.pdf"
    # add_pdf_to_db(pdf_path)

    # Query với full context pipeline
    client, child_vector_store = build_child_vector_store(
        collection_name=CHILD_COLLECTION,
        qdrant_path=QDRANT_PATH,
        recreate=False,
    )

    try:
        # k=20 để retrieve đủ candidates cho rerank + hybrid scoring
        child_retriever = build_retriever(child_vector_store, k=20)

        query = "cho tôi biết tóm tắt của phần cây xoài là gì"

        context_data = query_with_context(
            query=query,
            child_retriever=child_retriever,
            parent_store_path=PARENT_STORE_PATH,
        )

        print("===== MAIN CONTEXT =====")
        print(context_data["main_context"])
        print("\n===== SECONDARY CONTEXT =====")
        print(context_data["secondary_context"])
        print("\n===== FINAL CONTEXT FOR LLM =====")
        print(context_data["final_context"])

    finally:
        client.close()


if __name__ == "__main__":
    main()