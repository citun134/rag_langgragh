from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.qdrant import RetrievalMode

from ..embeddings.hf_embeddings import dense_embeddings

# Initialize Qdrant client (local file-based storage)
client = QdrantClient(path="qdrant_db")

# Get embedding dimension
embedding_dimension = len(dense_embeddings.embed_query("test"))

def ensure_collection(collection_name):
    """Create Qdrant collection if it doesn't exist"""
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=embedding_dimension,
                distance=qmodels.Distance.COSINE
            ),
            sparse_vectors_config={
                "sparse": qmodels.SparseVectorParams()
            },
        )
        print(f"✓ Created collection: {collection_name}")
    else:
        print(f"✓ Collection already exists: {collection_name}")

