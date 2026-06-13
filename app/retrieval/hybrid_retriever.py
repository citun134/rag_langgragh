from collections import defaultdict
from typing import List, Dict, Any
import re
import numpy as np
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore

RRF_K = 20


def _tokenize_for_bm25(text: str) -> List[str]:
    text = (text or "").lower()
    try:
        return word_tokenize(text)
    except LookupError:
        return re.findall(r"\w+", text, flags=re.UNICODE)


class SmartHybridRetriever(BaseRetriever):
    vector_store: QdrantVectorStore
    documents: List[Document]
    k: int = 12
    search_filter: Any = None

    class Config:
        arbitrary_types_allowed = True

    def _rrf(self, results_list: List[List[Document]]) -> List[Document]:
        rrf_scores = defaultdict(float)
        doc_map: Dict[str, Document] = {}

        for results in results_list:
            for rank, doc in enumerate(results, start=1):
                doc_id = doc.metadata.get("child_id") or doc.page_content[:80]
                rrf_scores[doc_id] += 1.0 / (RRF_K + rank)
                doc_map[doc_id] = doc

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_id] for doc_id, _ in sorted_docs]

    def model_post_init(self, __context):
        tokenized = [
            _tokenize_for_bm25(d.page_content or "")
            for d in self.documents
        ]
        self._bm25 = BM25Okapi(tokenized) if tokenized and any(tokenized) else None

    def _get_relevant_documents(self, query, *, run_manager=None):
        if self.search_filter is not None:
            vector_results = self.vector_store.similarity_search(
                query,
                k=self.k,
                filter=self.search_filter,
            )
        else:
            vector_results = self.vector_store.similarity_search(
                query,
                k=self.k,
            )

        bm25_results = []
        if self._bm25 is not None and self.documents:
            bm25_scores = self._bm25.get_scores(_tokenize_for_bm25(query))
            top_k_indices = np.argsort(bm25_scores)[::-1][:self.k]
            bm25_results = [self.documents[i] for i in top_k_indices]

        fused = self._rrf([vector_results, bm25_results])
        return fused[:self.k]
