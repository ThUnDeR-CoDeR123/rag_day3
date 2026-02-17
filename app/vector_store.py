import os
import chromadb
import uuid
import numpy as np
from typing import List, Any


class VectorStore:
    def __init__(self, collection_name="pdf_documents", persist_directory="data/vector_store"):
        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

        print(f"Vector store initialized. Documents count: {self.collection.count()}")

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        ids = []
        texts = []
        metadatas = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
            embeddings_list.append(embedding.tolist())

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings_list
        )

        print(f"Added {len(documents)} documents.")
