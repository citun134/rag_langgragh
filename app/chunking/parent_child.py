import os
import re
import glob
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.chunking.semantic_chunker import SemanticChunker
from app.embeddings.hf_embeddings import dense_embeddings
from app.utils.text_cleaning import clean_vietnamese_text
from app.config.settings import settings

# =========================
# CONFIG
# =========================
MAX_CHUNK_SIZE = settings.max_chunk_size
MIN_CHUNK_SIZE = settings.min_chunk_size
PARENT_BREAKPOINT_THRESHOLD = settings.parent_breakpoint_threshold
PARENT_OVERLAP_SIZE = settings.parent_overlap_size
CHILD_CHUNK_SIZE = settings.child_chunk_size
CHILD_CHUNK_OVERLAP = settings.child_chunk_overlap


def build_header_prefix(metadata: dict) -> str:
    parts = []
    for key in ["H1", "H2", "H3"]:
        value = metadata.get(key)
        if value:
            parts.append(str(value).strip())
    return " | ".join(parts)

def build_child_page_content(child_doc: Document) -> str:
    header_prefix = build_header_prefix(child_doc.metadata or {})
    text = (child_doc.page_content or "").strip()

    if header_prefix:
        return f"{header_prefix}\n\n{text}"
    return text

# =========================
# HELPERS
# =========================
def merge_tiny_parents(parents: List[Document], min_size: int = MIN_CHUNK_SIZE) -> List[Document]:
    if not parents:
        return []

    merged = []
    for doc in parents:
        text = (doc.page_content or "").strip()
        if not text:
            continue

        if merged and len(text) < min_size:
            merged[-1].page_content = merged[-1].page_content.rstrip() + "\n\n" + text
        else:
            merged.append(doc)

    return merged


def load_markdown_sections(markdown_dir: str) -> List[Document]:
    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    docs = []
    md_files = sorted(glob.glob(os.path.join(markdown_dir, "*.md")))

    if not md_files:
        raise ValueError(f"No .md files found in {markdown_dir}")

    for doc_path_str in md_files:
        doc_path = Path(doc_path_str)
        print(f"📄 Processing: {doc_path.name}")

        with open(doc_path, "r", encoding="utf-8") as f:
            md_text = clean_vietnamese_text(f.read())

        section_docs = header_splitter.split_text(md_text)
        if not section_docs:
            section_docs = [Document(page_content=md_text, metadata={})]

        for sec_idx, sec_doc in enumerate(section_docs):
            sec_doc.metadata = dict(sec_doc.metadata or {})
            sec_doc.metadata.update({
                "source": doc_path.stem + ".pdf",
                "source_md": doc_path.name,
                "section_index": sec_idx,
            })
            docs.append(sec_doc)

    return docs


def make_parent_child_docs(markdown_dir: str):
    base_docs = load_markdown_sections(markdown_dir)

    semantic_chunker = SemanticChunker(
        breakpoint_threshold=PARENT_BREAKPOINT_THRESHOLD,
        overlap_size=PARENT_OVERLAP_SIZE,
        embeddings=dense_embeddings,  # reuse loaded dense model
    )

    semantic_parents = semantic_chunker.split(base_docs)
    semantic_parents = merge_tiny_parents(semantic_parents, min_size=MIN_CHUNK_SIZE)

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    all_parent_pairs = []
    all_child_docs = []

    for parent_idx, parent_doc in enumerate(semantic_parents):
        source_stem = Path(parent_doc.metadata.get("source_md", "doc.md")).stem
        parent_id = f"{source_stem}_parent_{parent_idx:05d}"

        parent_meta = dict(parent_doc.metadata or {})
        parent_meta.update({
            "parent_id": parent_id,
            "retrieval_mode": "parent_semantic_child_recursive",
            "parent_length": len(parent_doc.page_content),
        })

        parent_doc.metadata = parent_meta
        all_parent_pairs.append((parent_id, parent_doc))

        child_docs = child_splitter.split_documents([parent_doc])

        for child_idx, child_doc in enumerate(child_docs):
            child_meta = dict(child_doc.metadata or {})
            child_meta.update({
                "parent_id": parent_id,
                "child_id": f"{parent_id}_child_{child_idx:03d}",
                "child_index": child_idx,
                "retrieval_mode": "parent_semantic_child_recursive",
                "child_length": len(child_doc.page_content),
                "header_prefix": build_header_prefix(child_doc.metadata or {})
            })
            child_doc.metadata = child_meta

            # rất quan trọng
            child_doc.page_content = build_child_page_content(child_doc)

            all_child_docs.append(child_doc)

    return all_parent_pairs, all_child_docs
