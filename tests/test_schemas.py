"""
Tests for RAG pipeline data models and schema validation.
"""

import pytest
from models.schemas import Document, RetrievedChunk, RAGResponse


def test_document_creation():
    doc = Document(
        content="AWS Bedrock is a fully managed service.",
        source="aws-docs.pdf",
        metadata={"page": 1},
    )
    assert doc.content == "AWS Bedrock is a fully managed service."
    assert doc.source == "aws-docs.pdf"
    assert doc.metadata["page"] == 1


def test_document_defaults():
    doc = Document(content="Some content")
    assert doc.source == ""
    assert doc.metadata == {}


def test_retrieved_chunk_creation():
    chunk = RetrievedChunk(
        content="Bedrock supports Claude, Titan, and Llama models.",
        score=0.92,
        source="bedrock-overview.pdf",
    )
    assert chunk.score == 0.92
    assert "Claude" in chunk.content


def test_rag_response_citations():
    chunks = [
        RetrievedChunk(content="chunk 1", score=0.9, source="doc1.pdf"),
        RetrievedChunk(content="chunk 2", score=0.8, source="doc2.pdf"),
    ]
    response = RAGResponse(
        answer="Bedrock is a managed AI service.",
        retrieved_chunks=chunks,
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    )
    sources = response.citations()
    assert "doc1.pdf" in sources
    assert "doc2.pdf" in sources


def test_rag_response_deduplicates_citations():
    chunks = [
        RetrievedChunk(content="chunk 1", score=0.9, source="doc1.pdf"),
        RetrievedChunk(content="chunk 2", score=0.8, source="doc1.pdf"),
    ]
    response = RAGResponse(
        answer="Answer here.",
        retrieved_chunks=chunks,
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    )
    assert response.citations().count("doc1.pdf") == 1
