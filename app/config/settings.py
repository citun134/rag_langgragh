from dataclasses import dataclass
import os

@dataclass
class Settings:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    embedding_model: str = "BAAI/bge-m3"
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name: str = "rag_docs"
    markdown_dir: str = "data/markdown"
    pdf_dir: str = "data/raw_pdfs"
    top_k_child: int = 12
    top_k_parent: int = 5
    max_context_tokens: int = 8000

settings = Settings()
