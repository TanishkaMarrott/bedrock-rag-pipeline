from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    content: str
    source: str = ""
    metadata: dict = Field(default_factory=dict)
    doc_id: Optional[str] = None
    title: Optional[str] = None


class RetrievedChunk(BaseModel):
    content: str
    score: float
    source: str = ""
    location: Optional[str] = None

    # backwards-compatible alias used internally
    @property
    def source_doc(self) -> str:
        return self.source


class RAGResponse(BaseModel):
    answer: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    model_id: str = ""
    query: str = ""
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    def citations(self) -> list[str]:
        return list({c.source for c in self.retrieved_chunks})
