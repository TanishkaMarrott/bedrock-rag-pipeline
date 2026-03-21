from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    title: str
    content: str
    source: str = ""
    metadata: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    content: str
    score: float
    source_doc: str
    location: Optional[str] = None


class RAGResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    model_id: str = ""
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    def citations(self) -> list[str]:
        return list({c.source_doc for c in self.retrieved_chunks})
