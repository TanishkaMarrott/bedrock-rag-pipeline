"""
Tests for BedrockRetriever in DEMO_MODE (no AWS credentials required).
"""

import os
import pytest

os.environ.setdefault("DEMO_MODE", "true")
os.environ.setdefault("AWS_REGION", "us-east-1")
os.environ.setdefault("KNOWLEDGE_BASE_ID", "demo-kb-id")
os.environ.setdefault("S3_BUCKET_NAME", "demo-bucket")

from rag.retriever import BedrockRetriever


def test_retriever_demo_mode_returns_response():
    retriever = BedrockRetriever()
    response = retriever.retrieve_and_generate("What is AWS Bedrock?")
    assert response.answer
    assert len(response.answer) > 0


def test_retriever_demo_mode_has_chunks():
    retriever = BedrockRetriever()
    response = retriever.retrieve_and_generate("Explain RAG pipelines.")
    assert len(response.retrieved_chunks) > 0


def test_retriever_demo_mode_chunks_have_scores():
    retriever = BedrockRetriever()
    response = retriever.retrieve_and_generate("What is Claude?")
    for chunk in response.retrieved_chunks:
        assert 0.0 <= chunk.score <= 1.0


def test_retriever_retrieve_only():
    retriever = BedrockRetriever()
    chunks = retriever.retrieve("Vector databases explained")
    assert isinstance(chunks, list)
    assert len(chunks) > 0
