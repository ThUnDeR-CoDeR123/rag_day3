from fastapi import FastAPI
from app.schemas import QueryRequest
from app.embeddings import EmbeddingManager
from app.vector_store import VectorStore
from app.retriever import RAGRetriever
from app.rag_engine import RAGEngine

app = FastAPI(title="Basic RAG API")

# Initialize components once
embedding_manager = EmbeddingManager()
vector_store = VectorStore()
retriever = RAGRetriever(vector_store, embedding_manager)
rag_engine = RAGEngine(retriever)


@app.post("/query")
async def query_rag(request: QueryRequest):
    result = rag_engine.generate_answer(
        request.question,
        top_k=request.top_k
    )
    return result
