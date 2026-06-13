import json
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_core.documents import Document

from app.security.access_control import (
    ROLE_ALLOWED_VISIBILITIES,
    VALID_VISIBILITIES,
    normalize_role,
)


DEFAULT_CHILD_DOCS_PATH = Path("storage/child_docs.jsonl")


def _doc_key(doc: Document) -> str:
    metadata = doc.metadata or {}
    child_id = str(metadata.get("child_id") or "").strip()
    if child_id:
        return child_id

    parent_id = str(metadata.get("parent_id") or "").strip()
    child_index = str(metadata.get("child_index") or "").strip()
    if parent_id or child_index:
        return f"{parent_id}:{child_index}"

    return (doc.page_content or "")[:200]


def dedupe_documents(documents: Iterable[Document]) -> List[Document]:
    seen = set()
    unique = []

    for doc in documents:
        key = _doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)

    return unique


def save_child_documents(
    documents: Iterable[Document],
    path: str | Path = DEFAULT_CHILD_DOCS_PATH,
    append: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for doc in documents:
            payload = {
                "page_content": doc.page_content or "",
                "metadata": doc.metadata or {},
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_child_documents(path: str | Path = DEFAULT_CHILD_DOCS_PATH) -> List[Document]:
    path = Path(path)
    if not path.exists():
        return []

    docs: List[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            docs.append(
                Document(
                    page_content=item.get("page_content", "") or "",
                    metadata=item.get("metadata", {}) or {},
                )
            )

    return dedupe_documents(docs)


def _payload_to_document(payload: dict) -> Optional[Document]:
    payload = dict(payload or {})

    page_content = (
        payload.get("page_content")
        or payload.get("content")
        or payload.get("text")
        or ""
    )

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    # Keep useful top-level payload fields if this collection was not written
    # by LangChain's default page_content/metadata payload layout.
    for key, value in payload.items():
        if key in {"page_content", "content", "text", "metadata"}:
            continue
        metadata.setdefault(key, value)

    if not page_content:
        return None

    return Document(page_content=str(page_content), metadata=metadata)


def load_child_documents_from_qdrant(
    client,
    collection_name: str,
    scroll_filter=None,
    batch_size: int = 256,
    limit: int | None = None,
) -> List[Document]:
    docs: List[Document] = []
    offset = None

    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for record in records:
            doc = _payload_to_document(getattr(record, "payload", {}) or {})
            if doc is not None:
                docs.append(doc)

            if limit is not None and len(docs) >= limit:
                return dedupe_documents(docs[:limit])

        if offset is None:
            break

    return dedupe_documents(docs)


def filter_documents_by_access(
    documents: Iterable[Document],
    user_id: str = "anonymous",
    role: str = "employee",
) -> List[Document]:
    role = normalize_role(role)
    if role == "admin":
        return list(documents)

    user_id = str(user_id or "anonymous").strip()
    allowed_visibilities = set(ROLE_ALLOWED_VISIBILITIES[role])

    filtered = []
    for doc in documents:
        metadata = doc.metadata or {}
        visibility = str(metadata.get("visibility") or "public").lower().strip()
        owner_user_id = str(metadata.get("owner_user_id") or "").strip()

        if visibility not in VALID_VISIBILITIES:
            visibility = "public"

        if visibility in allowed_visibilities:
            filtered.append(doc)
        elif visibility == "private" and owner_user_id == user_id:
            filtered.append(doc)

    return filtered
