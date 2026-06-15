from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse

from app.config.settings import settings

dense_embeddings = HuggingFaceEmbeddings(
    model_name=settings.dense_embedding_model,
)

sparse_embeddings = FastEmbedSparse(
    model_name=settings.sparse_embedding_model,
)