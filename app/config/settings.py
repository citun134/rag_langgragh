# app/config/settings.py

from dataclasses import dataclass
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    # =========================
    # LLM
    # =========================
    llm_provider: str = os.getenv("LLM_PROVIDER", "kaggle_api")  # kaggle_api | local_hf

    llm_model_name: str = os.getenv(
        "LLM_MODEL_NAME",
        "Qwen/Qwen2.5-0.5B-Instruct",
    )

    kaggle_api_url: str = os.getenv(
        "KAGGLE_API_URL",
        "https://ecwvf-35-186-160-201.run.pinggy-free.link",
    )

    kaggle_api_key: str = os.getenv(
        "KAGGLE_API_KEY",
        "my-secret-api-key-123",
    )

    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    llm_max_new_tokens: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "300"))

    # =========================
    # Embeddings
    # =========================
    dense_embedding_model: str = os.getenv(
        "DENSE_EMBEDDING_MODEL",
        "AITeamVN/Vietnamese_Embedding_v2",
    )

    sparse_embedding_model: str = os.getenv(
        "SPARSE_EMBEDDING_MODEL",
        "Qdrant/bm25",
    )

    # =========================
    # Qdrant / Storage
    # =========================
    qdrant_path: str = os.getenv("QDRANT_PATH", "qdrant_db")
    child_collection: str = os.getenv(
        "CHILD_COLLECTION",
        "document_child_chunks",
    )

    pdf_dir: str = os.getenv("PDF_DIR", str(BASE_DIR / "data"))
    markdown_dir: str = os.getenv("MARKDOWN_DIR", str(BASE_DIR / "markdown"))
    parent_store_path: str = os.getenv(
        "PARENT_STORE_PATH",
        str(BASE_DIR / "parent_store"),
    )
    child_docs_path: str = os.getenv(
        "CHILD_DOCS_PATH",
        "storage/child_docs.jsonl",
    )

    # =========================
    # Chunking
    # =========================
    max_chunk_size: int = int(os.getenv("MAX_CHUNK_SIZE", "2200"))
    min_chunk_size: int = int(os.getenv("MIN_CHUNK_SIZE", "600"))
    parent_breakpoint_threshold: float = float(
        os.getenv("PARENT_BREAKPOINT_THRESHOLD", "0.68")
    )
    parent_overlap_size: int = int(os.getenv("PARENT_OVERLAP_SIZE", "120"))
    child_chunk_size: int = int(os.getenv("CHILD_CHUNK_SIZE", "500"))
    child_chunk_overlap: int = int(os.getenv("CHILD_CHUNK_OVERLAP", "100"))

    # =========================
    # Retrieval / Rerank
    # =========================
    top_k_child: int = int(os.getenv("TOP_K_CHILD", "30"))
    top_k_child_summary: int = int(os.getenv("TOP_K_CHILD_SUMMARY", "30"))
    top_k_child_enum: int = int(os.getenv("TOP_K_CHILD_ENUM", "30"))

    top_k_parent: int = int(os.getenv("TOP_K_PARENT", "3"))
    top_k_parent_summary: int = int(os.getenv("TOP_K_PARENT_SUMMARY", "5"))
    top_k_parent_enum: int = int(os.getenv("TOP_K_PARENT_ENUM", "4"))

    rank_constant: int = int(os.getenv("RANK_CONSTANT", "20"))

    reranker_model: str = os.getenv(
        "RERANKER_MODEL",
        "AITeamVN/Vietnamese_Reranker",
    )
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "25"))
    rerank_batch_size: int = int(os.getenv("RERANK_BATCH_SIZE", "16"))

    # =========================
    # Context
    # =========================
    max_parent_chars: int = int(os.getenv("MAX_PARENT_CHARS", "1600"))
    max_child_chars: int = int(os.getenv("MAX_CHILD_CHARS", "400"))
    max_secondary_parents: int = int(os.getenv("MAX_SECONDARY_PARENTS", "1"))
    secondary_ratio_threshold: float = float(
        os.getenv("SECONDARY_RATIO_THRESHOLD", "0.55")
    )


settings = Settings()