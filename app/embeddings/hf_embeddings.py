from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse

# Dense embeddings for semantic understanding
dense_embeddings = HuggingFaceEmbeddings(
    # model_name="sentence-transformers/all-mpnet-base-v2"
    # model_name="bkai-foundation-models/vietnamese-bi-encoder"
    model_name="AITeamVN/Vietnamese_Embedding_v2"

)

# Sparse embeddings for keyword matching
sparse_embeddings = FastEmbedSparse(
    model_name="Qdrant/bm25"
)