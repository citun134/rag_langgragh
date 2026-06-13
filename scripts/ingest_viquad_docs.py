import sys
from pathlib import Path

# Cho phép chạy trực tiếp bằng:
# python scripts/ingest_viquad_docs.py --markdown-dir data/eval/viquad_docs
# Khi chạy trực tiếp, sys.path mặc định trỏ vào thư mục scripts/,
# nên cần thêm project root vào sys.path để import được app.*
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import shutil
from pathlib import Path

from app.chunking.parent_child import make_parent_child_docs
from app.embeddings.hf_embeddings import dense_embeddings, sparse_embeddings
from app.ingestion.indexing import (
    CHILD_COLLECTION,
    QDRANT_PATH,
    build_child_vector_store,
    save_parent_store,
    attach_access_metadata,
)
from app.ingestion.pdf_to_markdown import PARENT_STORE_PATH
from app.retrieval.child_documents import DEFAULT_CHILD_DOCS_PATH, save_child_documents
from app.security.access_control import build_document_metadata


def _read_markdown_files(markdown_dir: Path) -> dict[str, str]:
    """
    Đọc các file markdown để fallback đoán file_name nếu chunk metadata chưa có source.
    """
    files = {}
    for path in sorted(markdown_dir.glob("*.md")):
        try:
            files[path.name] = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            files[path.name] = ""
    return files


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _guess_file_name(doc, markdown_texts: dict[str, str]) -> str | None:
    """
    Ưu tiên lấy filename từ metadata source/file_name/filename.
    Nếu không có thì thử dò theo nội dung chunk trong các file .md.
    """
    metadata = dict(getattr(doc, "metadata", {}) or {})

    for key in ("file_name", "filename", "source_md", "source", "path", "file_path"):
        value = metadata.get(key)
        if value:
            return Path(str(value)).name

    content = _normalize_text(getattr(doc, "page_content", "") or "")
    if not content:
        return None

    # Lấy snippet đủ dài để dò trong markdown gốc.
    snippet = content[:300]
    if len(snippet) < 50:
        return None

    for file_name, full_text in markdown_texts.items():
        if snippet in _normalize_text(full_text):
            return file_name

    return None


def _attach_file_name_metadata(all_parent_pairs, all_child_docs, markdown_dir: Path):
    """
    Đảm bảo parent/child metadata có file_name và filename.
    Điều này giúp rag_evaluate.py chấm doc_hit_at_k đúng.
    """
    markdown_texts = _read_markdown_files(markdown_dir)

    for parent_id, parent_doc in all_parent_pairs:
        parent_doc.metadata = dict(parent_doc.metadata or {})
        file_name = _guess_file_name(parent_doc, markdown_texts)

        if file_name:
            parent_doc.metadata["file_name"] = file_name
            parent_doc.metadata["filename"] = file_name
            parent_doc.metadata.setdefault("doc_id", file_name)

        parent_doc.metadata["parent_id"] = parent_id

    for child_doc in all_child_docs:
        child_doc.metadata = dict(child_doc.metadata or {})
        file_name = _guess_file_name(child_doc, markdown_texts)

        if file_name:
            child_doc.metadata["file_name"] = file_name
            child_doc.metadata["filename"] = file_name
            child_doc.metadata.setdefault("doc_id", file_name)

    return all_parent_pairs, all_child_docs


def _append_parent_store(all_parent_pairs, parent_store_path: Path):
    """
    Append parent chunks vào parent_store, không xóa dữ liệu cũ.
    """
    parent_store_path.mkdir(parents=True, exist_ok=True)

    for parent_id, parent_doc in all_parent_pairs:
        filepath = parent_store_path / f"{parent_id}.json"
        payload = {
            "page_content": parent_doc.page_content,
            "metadata": parent_doc.metadata,
        }
        filepath.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def ingest_markdown_docs(
    markdown_dir: str | Path = "data/eval/viquad_docs",
    parent_store_path: str | Path = PARENT_STORE_PATH,
    collection_name: str = CHILD_COLLECTION,
    qdrant_path: str = QDRANT_PATH,
    recreate: bool = True,
    visibility: str = "public",
    owner_user_id: str = "admin",
):
    markdown_dir = Path(markdown_dir)
    parent_store_path = Path(parent_store_path)

    if not markdown_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy markdown_dir: {markdown_dir}")

    md_files = sorted(markdown_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"Không có file .md nào trong: {markdown_dir}")

    print(f"📁 Markdown dir: {markdown_dir}")
    print(f"📄 Markdown files: {len(md_files)}")
    print(f"🧠 Qdrant path: {qdrant_path}")
    print(f"📦 Collection: {collection_name}")
    print(f"💾 Parent store: {parent_store_path}")
    print(f"♻️ Recreate: {recreate}")

    client, child_vector_store = build_child_vector_store(
        collection_name=collection_name,
        qdrant_path=qdrant_path,
        recreate=recreate,
    )

    try:
        print("\n🔪 Chunking markdown docs...")
        all_parent_pairs, all_child_docs = make_parent_child_docs(markdown_dir)

        if not all_parent_pairs or not all_child_docs:
            print("⚠️ Không tạo được parent/child chunks.")
            return

        print(f"✓ Parent chunks: {len(all_parent_pairs)}")
        print(f"✓ Child chunks:  {len(all_child_docs)}")

        print("\n🏷️ Attaching access metadata...")
        access_metadata = build_document_metadata(
            visibility=visibility,
            owner_user_id=owner_user_id,
        )
        all_parent_pairs, all_child_docs = attach_access_metadata(
            all_parent_pairs=all_parent_pairs,
            all_child_docs=all_child_docs,
            access_metadata=access_metadata,
        )

        print("🏷️ Attaching file_name metadata for evaluation...")
        all_parent_pairs, all_child_docs = _attach_file_name_metadata(
            all_parent_pairs=all_parent_pairs,
            all_child_docs=all_child_docs,
            markdown_dir=markdown_dir,
        )

        print("\n🔍 Indexing child chunks into Qdrant...")
        child_vector_store.add_documents(all_child_docs)
        print("✓ Child chunks indexed.")

        print("\n💾 Saving child docs cache...")
        save_child_documents(
            all_child_docs,
            DEFAULT_CHILD_DOCS_PATH,
            append=not recreate,
        )
        print(f"✓ Child docs saved to: {DEFAULT_CHILD_DOCS_PATH}")

        print("\n💾 Saving parent store...")
        if recreate:
            save_parent_store(all_parent_pairs, parent_store_path)
        else:
            _append_parent_store(all_parent_pairs, parent_store_path)
            print("✓ Parent store appended successfully.")

        print("\n✅ Ingest hoàn tất.")
        print("Bạn có thể chạy rag_evaluate.py với dataset ViQuAD JSONL ngay bây giờ.")

    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Ingest UIT-ViQuAD markdown docs into the existing RAG Qdrant + parent store."
    )
    parser.add_argument(
        "--markdown-dir",
        default="data/eval/viquad_docs",
        help="Folder chứa các file .md đã convert từ UIT-ViQuAD2.0.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append vào DB hiện tại. Mặc định là recreate để benchmark sạch.",
    )
    parser.add_argument(
        "--visibility",
        default="public",
        choices=["public", "private"],
        help="Document visibility metadata.",
    )
    parser.add_argument(
        "--owner-user-id",
        default="admin",
        help="Owner user id metadata.",
    )
    args = parser.parse_args()

    ingest_markdown_docs(
        markdown_dir=args.markdown_dir,
        recreate=not args.append,
        visibility=args.visibility,
        owner_user_id=args.owner_user_id,
    )


if __name__ == "__main__":
    main()
