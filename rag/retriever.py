"""
Bedrock RAG Retriever

Two retrieval modes:

  retrieve_and_generate()  — single API call: retrieves relevant chunks
                             AND generates a grounded answer using Claude.
                             Bedrock handles the full RAG loop internally.

  retrieve()               — retrieves chunks only, no generation.
                             Use when you want to control generation yourself.

In DEMO_MODE both return realistic mock responses so the pipeline
runs end-to-end without real AWS credentials or a live Knowledge Base.
"""

from __future__ import annotations

import json
import os

from botocore.exceptions import ClientError
from rich.console import Console

from models.schemas import RAGResponse, RetrievedChunk
from rag.bedrock_client import BedrockClient

console = Console()

DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
KB_ID = os.getenv("BEDROCK_KB_ID", "demo-kb-id")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")

# Realistic mock data for demo mode
_MOCK_CHUNKS = [
    RetrievedChunk(
        content="AWS Bedrock provides fully managed access to foundation models. "
                "It supports Claude, Titan, Llama, and other models via a single API.",
        score=0.94,
        source_doc="bedrock-overview.txt",
        location="paragraph 1",
    ),
    RetrievedChunk(
        content="Bedrock Knowledge Bases automate the RAG pipeline: chunking, embedding, "
                "and indexing documents into a managed vector store (OpenSearch Serverless).",
        score=0.89,
        source_doc="bedrock-kb-guide.txt",
        location="paragraph 3",
    ),
    RetrievedChunk(
        content="The RetrieveAndGenerate API retrieves relevant passages and passes them "
                "to the foundation model as context, producing grounded, citation-backed answers.",
        score=0.85,
        source_doc="bedrock-api-reference.txt",
        location="section 4.2",
    ),
]

_MOCK_ANSWERS = {
    "default": (
        "AWS Bedrock is a fully managed service that provides access to foundation models "
        "from Anthropic, Amazon, Meta, and others. Its Knowledge Base feature automates the "
        "RAG pipeline by handling document chunking, embedding, and vector storage. "
        "The RetrieveAndGenerate API lets you query your documents and get grounded answers "
        "with citations in a single API call."
    )
}


class BedrockRetriever:
    def __init__(self, client: BedrockClient) -> None:
        self.client = client

    def retrieve_and_generate(self, query: str) -> RAGResponse:
        """
        Single Bedrock API call that:
          1. Embeds the query using the KB's embedding model
          2. Retrieves top-k relevant chunks from the vector store
          3. Passes chunks + query to Claude as context
          4. Returns a grounded answer with source citations

        This is the primary RAG entry point.
        """
        if DEMO_MODE:
            return self._mock_response(query)

        try:
            response = self.client.agent_runtime.retrieve_and_generate(
                input={"text": query},
                retrieveAndGenerateConfiguration={
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": KB_ID,
                        "modelArn": f"arn:aws:bedrock:{self.client.region}::foundation-model/{MODEL_ID}",
                        "retrievalConfiguration": {
                            "vectorSearchConfiguration": {
                                "numberOfResults": 5,
                            }
                        },
                    },
                },
            )

            answer = response["output"]["text"]
            chunks = []
            for citation in response.get("citations", []):
                for ref in citation.get("retrievedReferences", []):
                    chunks.append(RetrievedChunk(
                        content=ref["content"]["text"],
                        score=ref.get("score", 0.0),
                        source_doc=ref.get("location", {}).get("s3Location", {}).get("uri", "unknown"),
                        location=str(ref.get("location", "")),
                    ))

            return RAGResponse(
                query=query,
                answer=answer,
                retrieved_chunks=chunks,
                model_id=MODEL_ID,
            )

        except ClientError as e:
            console.print(f"[red]Bedrock error:[/red] {e}")
            raise

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve chunks only — no generation."""
        if DEMO_MODE:
            return _MOCK_CHUNKS[:top_k]

        try:
            response = self.client.agent_runtime.retrieve(
                knowledgeBaseId=KB_ID,
                retrievalQuery={"text": query},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {"numberOfResults": top_k}
                },
            )
            return [
                RetrievedChunk(
                    content=r["content"]["text"],
                    score=r.get("score", 0.0),
                    source_doc=r.get("location", {}).get("s3Location", {}).get("uri", ""),
                )
                for r in response.get("retrievalResults", [])
            ]
        except ClientError as e:
            console.print(f"[red]Retrieve error:[/red] {e}")
            raise

    def _mock_response(self, query: str) -> RAGResponse:
        answer = _MOCK_ANSWERS.get(query.lower(), _MOCK_ANSWERS["default"])
        return RAGResponse(
            query=query,
            answer=answer,
            retrieved_chunks=_MOCK_CHUNKS,
            model_id=MODEL_ID,
        )
