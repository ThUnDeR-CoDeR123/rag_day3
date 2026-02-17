from app.loader import process_all_pdfs, split_documents
from app.embeddings import EmbeddingManager
from app.vector_store import VectorStore

docs = process_all_pdfs("data/")
chunks = split_documents(docs)

embedding_manager = EmbeddingManager()
vector_store = VectorStore()

texts = [doc.page_content for doc in chunks]
embeddings = embedding_manager.generate_embeddings(texts)

vector_store.add_documents(chunks, embeddings)
