from typing import List, Dict, Any


class RAGRetriever:
    def __init__(self, vector_store, embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        retrieved_docs = []

        if results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            for i, (doc_id, doc, meta, distance) in enumerate(zip(ids, docs, metadatas, distances)):
                similarity_score = 1 - distance
                retrieved_docs.append({
                    "id": doc_id,
                    "content": doc,
                    "metadata": meta,
                    "similarity_score": similarity_score,
                    "rank": i + 1
                })

        return retrieved_docs
