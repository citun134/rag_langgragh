import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.qdrant import RetrievalMode

from app.ingestion.pdf_to_markdown import MARKDOWN_DIR, PARENT_STORE_PATH, pdf_to_markdown
from app.chunking.parent_child import make_parent_child_docs
from app.embeddings.hf_embeddings import dense_embeddings, sparse_embeddings


CHILD_COLLECTION = "document_child_chunks"
QDRANT_PATH = "qdrant_db"


def build_child_vector_store(
    collection_name: str = CHILD_COLLECTION,
    qdrant_path: str = QDRANT_PATH,
    recreate: bool = False,
):
    """
    Build Qdrant client and child vector store.

    Returns:
        client: QdrantClient
        child_vector_store: QdrantVectorStore
    """
    client = QdrantClient(path=qdrant_path)

    embedding_dimension = len(dense_embeddings.embed_query("test"))

    if recreate and client.collection_exists(collection_name):
        print(f"Removing existing Qdrant collection: {collection_name}")
        client.delete_collection(collection_name)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=embedding_dimension,
                distance=qmodels.Distance.COSINE,
            ),
            sparse_vectors_config={
                "sparse": qmodels.SparseVectorParams()
            },
        )
        print(f"✓ Created collection: {collection_name}")
    else:
        print(f"✓ Collection already exists: {collection_name}")

    child_vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name="sparse",
    )

    return client, child_vector_store


def save_parent_store(all_parent_pairs, parent_store_path: Path | str = PARENT_STORE_PATH):
    parent_store_path = Path(parent_store_path)
    parent_store_path.mkdir(parents=True, exist_ok=True)

    for item in parent_store_path.iterdir():
        if item.is_file():
            item.unlink()

    for parent_id, parent_doc in all_parent_pairs:
        filepath = parent_store_path / f"{parent_id}.json"

        payload = {
            "page_content": parent_doc.page_content,
            "metadata": parent_doc.metadata,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print("✓ Parent store saved successfully.")


def index_documents_semantic_parent_child(
    markdown_dir: Path | str = MARKDOWN_DIR,
    parent_store_path: Path | str = PARENT_STORE_PATH,
    collection_name: str = CHILD_COLLECTION,
    qdrant_path: str = QDRANT_PATH,
    recreate_collection: bool = True,
):
    """
    Create child vector store, index child chunks, and save parent chunks.
    """
    client, child_vector_store = build_child_vector_store(
        collection_name=collection_name,
        qdrant_path=qdrant_path,
        recreate=recreate_collection,
    )

    try:
        all_parent_pairs, all_child_docs = make_parent_child_docs(markdown_dir)

        if not all_child_docs:
            print("⚠️ No child chunks to index.")
            return child_vector_store
            # return

        print(f"🔍 Indexing {len(all_child_docs)} child chunks into Qdrant...")
        child_vector_store.add_documents(all_child_docs)
        print("✓ Child chunks indexed successfully.")

        print(f"💾 Saving {len(all_parent_pairs)} parent chunks to JSON...")
        save_parent_store(all_parent_pairs, parent_store_path)

        return child_vector_store

    finally:
        client.close()



# =========================== #
# ====== ADD PDF TO DB ====== #
# =========================== #

def add_pdf_to_db(
    pdf_path: str | Path,
    markdown_dir: Path | str = MARKDOWN_DIR,
    parent_store_path: Path | str = PARENT_STORE_PATH,
    collection_name: str = CHILD_COLLECTION,
    qdrant_path: str = QDRANT_PATH,
):
    """
    Thêm một file PDF vào DB:
    1. Convert PDF → Markdown (chỉ file đó)
    2. Chunk theo parent-child (chỉ file mới)
    3. Index child chunks vào Qdrant (append, không recreate)
    4. Lưu parent chunks vào JSON store
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File không tồn tại: {pdf_path}")

    print(f"\n📄 Đang xử lý: {pdf_path.name}")

    # Bước 1: PDF → Markdown (chỉ file này)
    md_path = pdf_to_markdown(pdf_path, output_dir=markdown_dir)
    print(f"✓ Đã convert: {md_path.name}")

    # Bước 2: Build vector store (không recreate — append vào collection cũ)
    client, child_vector_store = build_child_vector_store(
        collection_name=collection_name,
        qdrant_path=qdrant_path,
        recreate=False,  # ← quan trọng: không xóa data cũ
    )

    try:
        # Bước 3: Chunk chỉ file mới này
        # Tạo temp markdown dir chứa đúng 1 file
        import tempfile, shutil
        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.copy(md_path, Path(tmp_dir) / md_path.name)
            all_parent_pairs, all_child_docs = make_parent_child_docs(tmp_dir)

        if not all_child_docs:
            print("⚠️ Không có child chunks nào được tạo.")
            return

        # Bước 4: Index child chunks
        print(f"🔍 Indexing {len(all_child_docs)} child chunks...")
        child_vector_store.add_documents(all_child_docs)
        print("✓ Child chunks đã được index.")

        # Bước 5: Lưu parent chunks (append vào parent store)
        parent_store_path = Path(parent_store_path)
        parent_store_path.mkdir(parents=True, exist_ok=True)

        for parent_id, parent_doc in all_parent_pairs:
            filepath = parent_store_path / f"{parent_id}.json"
            payload = {
                "page_content": parent_doc.page_content,
                "metadata": parent_doc.metadata,
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"✓ Đã lưu {len(all_parent_pairs)} parent chunks.")
        print(f"\n✅ Hoàn thành: {pdf_path.name} → DB")

    finally:
        client.close()

if __name__ == "__main__":
    index_documents_semantic_parent_child()