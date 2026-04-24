import numpy as np
from tqdm import tqdm
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.utils.text_cleaning import clean_vietnamese_text

# =========================
# CONFIG
# =========================
MAX_CHUNK_SIZE = 2200
MIN_CHUNK_SIZE = 600

# =========================
# SEMANTIC CHUNKER
# =========================
class SemanticChunker:
    def __init__(
            self,
            embedding_model: str = "bkai-foundation-models/vietnamese-bi-encoder",
            breakpoint_threshold: float = 0.5,
            overlap_size: int = 120,
            use_model_openai: bool = False,
            embeddings=None,  # optional: reuse already-loaded embeddings
    ):
        self.breakpoint_threshold = breakpoint_threshold
        self.overlap_size = overlap_size

        if embeddings is not None:
            self.embeddings = embeddings
            print("Using shared embedding model...")
        else:
            print(f"Loading embedding model: {embedding_model}...")
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

            # if use_model_openai:
            #     self.embeddings = OpenAIEmbeddings(
            #         model=embedding_model,
            #         api_key=OPENAI_API_KEY
            #     )
            # else:
            #     self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    def _split_into_sentences(self, text: str) -> List[str]:
        # Tách theo dấu câu tiếng Việt, không dùng NLTK
        import re
        # Tách theo . ! ? nhưng tránh viết tắt số thứ tự (1. 2. A. B.)
        parts = re.split(r'(?<=[^0-9A-Z])[.!?]\s+(?=[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠ-Ỹ])', text)
        result = []
        for p in parts:
            p = p.strip()
            if p and len(p) > 20:
                result.append(p)
        return result

    def _calculate_cosine_similarity(self, emb1, emb2):
        emb1 = np.asarray(emb1, dtype=np.float32)
        emb2 = np.asarray(emb2, dtype=np.float32)
        denom = (np.linalg.norm(emb1) * np.linalg.norm(emb2)) + 1e-12
        return float(np.dot(emb1, emb2) / denom)

    def _chunk_by_semantic_similarity(self, sentences: List[str]) -> List[str]:
        if not sentences:
            return []

        sentence_embeddings = self.embeddings.embed_documents(sentences)
        sentence_embeddings = [np.asarray(e, dtype=np.float32) for e in sentence_embeddings]

        chunks = []
        current_chunk = [sentences[0]]
        current_embs = [sentence_embeddings[0]]

        for i in range(1, len(sentences)):
            curr_sent = sentences[i]
            curr_emb = sentence_embeddings[i]

            centroid = np.mean(current_embs, axis=0)
            similarity = self._calculate_cosine_similarity(centroid, curr_emb)

            chunk_text = " ".join(current_chunk)
            chunk_len = len(chunk_text)

            if chunk_len >= MAX_CHUNK_SIZE:
                chunks.append(chunk_text)
                current_chunk = [curr_sent]
                current_embs = [curr_emb]
            elif similarity >= self.breakpoint_threshold or chunk_len < MIN_CHUNK_SIZE:
                current_chunk.append(curr_sent)
                current_embs.append(curr_emb)
            else:
                chunks.append(chunk_text)
                current_chunk = [curr_sent]
                current_embs = [curr_emb]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def split(self, documents: List[Document]) -> List[Document]:
        all_chunks = []

        for doc in tqdm(documents, desc="Semantic chunking parents"):
            text = clean_vietnamese_text(doc.page_content)
            sentences = self._split_into_sentences(text)

            if not sentences:
                if text.strip():
                    all_chunks.append(
                        Document(
                            page_content=text,
                            metadata=doc.metadata.copy(),
                        )
                    )
                continue

            chunks = self._chunk_by_semantic_similarity(sentences)

            for idx, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue

                # if idx > 0 and self.overlap_size > 0:
                #     prev_chunk = chunks[idx - 1]
                #     overlap = prev_chunk[-self.overlap_size:] if len(prev_chunk) >= self.overlap_size else prev_chunk
                #     chunk_text = overlap + " " + chunk_text

                chunk_doc = Document(
                    page_content=chunk_text.strip(),
                    metadata=doc.metadata.copy()
                )
                all_chunks.append(chunk_doc)

        return all_chunks