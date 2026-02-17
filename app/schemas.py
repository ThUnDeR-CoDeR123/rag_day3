from pydantic import BaseModel
from typing import List, Dict

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
