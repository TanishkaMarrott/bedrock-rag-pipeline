"""
Bedrock Client

Wraps three boto3 clients used across the pipeline:

  bedrock-agent-runtime  — RetrieveAndGenerate (RAG query)
                         — Retrieve (retrieval only, no generation)

  bedrock-runtime        — InvokeModel (direct Claude calls)

  bedrock-agent          — CreateKnowledgeBase, StartIngestionJob (setup)

All three are scoped to the same AWS region.
"""

from __future__ import annotations

import os
import boto3

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
EMBED_MODEL_ID = os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")


class BedrockClient:
    def __init__(self) -> None:
        self.region = REGION
        self.model_id = MODEL_ID
        self.embed_model_id = EMBED_MODEL_ID

        # Knowledge Base query + RAG
        self.agent_runtime = boto3.client(
            "bedrock-agent-runtime", region_name=REGION
        )
        # Direct model invocation
        self.runtime = boto3.client(
            "bedrock-runtime", region_name=REGION
        )
        # Knowledge Base management
        self.agent = boto3.client(
            "bedrock-agent", region_name=REGION
        )
        # S3 for document storage
        self.s3 = boto3.client("s3", region_name=REGION)
