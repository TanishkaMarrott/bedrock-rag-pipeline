"""
Bedrock Knowledge Base

Handles document ingestion into AWS Bedrock Knowledge Base.

Flow:
  1. Upload documents to S3
  2. Create Knowledge Base (if not exists) — Bedrock manages
     the vector store (OpenSearch Serverless) and embedding model
  3. Start ingestion job — Bedrock chunks, embeds, and indexes docs
  4. Poll until ingestion completes

In DEMO_MODE the ingestion is simulated — no real AWS calls.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from botocore.exceptions import ClientError
from rich.console import Console

from models.schemas import Document
from rag.bedrock_client import BedrockClient

console = Console()

DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
S3_BUCKET = os.getenv("BEDROCK_KB_S3_BUCKET", "my-bedrock-kb-docs")
KB_ID = os.getenv("BEDROCK_KB_ID", "")


class KnowledgeBaseManager:
    def __init__(self, client: BedrockClient) -> None:
        self.client = client

    def upload_documents(self, documents: list[Document]) -> list[str]:
        """Upload documents to S3 for Bedrock ingestion."""
        if DEMO_MODE:
            console.print(f"  [dim][DEMO] Simulating upload of {len(documents)} documents to S3[/dim]")
            return [f"s3://{S3_BUCKET}/{doc.doc_id}.txt" for doc in documents]

        uploaded = []
        for doc in documents:
            key = f"docs/{doc.doc_id}.txt"
            body = f"Title: {doc.title}\n\n{doc.content}"
            try:
                self.client.s3.put_object(
                    Bucket=S3_BUCKET,
                    Key=key,
                    Body=body.encode("utf-8"),
                    ContentType="text/plain",
                )
                s3_uri = f"s3://{S3_BUCKET}/{key}"
                uploaded.append(s3_uri)
                console.print(f"  [green]✓[/green] Uploaded: {doc.title}")
            except ClientError as e:
                console.print(f"  [red]✗[/red] Failed to upload {doc.title}: {e}")

        return uploaded

    def start_ingestion(self, kb_id: str) -> str:
        """Start a Bedrock Knowledge Base ingestion job."""
        if DEMO_MODE:
            console.print("  [dim][DEMO] Simulating Bedrock ingestion job[/dim]")
            return "demo-ingestion-job-id"

        try:
            # Get the data source for this KB
            data_sources = self.client.agent.list_data_sources(
                knowledgeBaseId=kb_id
            )
            data_source_id = data_sources["dataSourceSummaries"][0]["dataSourceId"]

            response = self.client.agent.start_ingestion_job(
                knowledgeBaseId=kb_id,
                dataSourceId=data_source_id,
            )
            job_id = response["ingestionJob"]["ingestionJobId"]
            console.print(f"  [cyan]Started ingestion job:[/cyan] {job_id}")
            return job_id
        except ClientError as e:
            console.print(f"  [red]Ingestion failed:[/red] {e}")
            raise

    def wait_for_ingestion(self, kb_id: str, job_id: str, timeout: int = 300) -> bool:
        """Poll until ingestion job completes or times out."""
        if DEMO_MODE:
            console.print("  [dim][DEMO] Ingestion complete (simulated)[/dim]")
            return True

        data_sources = self.client.agent.list_data_sources(knowledgeBaseId=kb_id)
        data_source_id = data_sources["dataSourceSummaries"][0]["dataSourceId"]

        start = time.time()
        while time.time() - start < timeout:
            response = self.client.agent.get_ingestion_job(
                knowledgeBaseId=kb_id,
                dataSourceId=data_source_id,
                ingestionJobId=job_id,
            )
            status = response["ingestionJob"]["status"]
            if status == "COMPLETE":
                console.print("  [green]✓[/green] Ingestion complete")
                return True
            if status == "FAILED":
                console.print(f"  [red]✗[/red] Ingestion failed: {response['ingestionJob']}")
                return False
            console.print(f"  [dim]Ingestion status: {status}...[/dim]")
            time.sleep(10)

        console.print("[yellow]Ingestion timed out[/yellow]")
        return False

    def ingest(self, documents: list[Document]) -> bool:
        """Full ingestion flow: upload → ingest → wait."""
        kb_id = KB_ID or os.getenv("BEDROCK_KB_ID", "")
        if not kb_id and not DEMO_MODE:
            raise ValueError("BEDROCK_KB_ID not set. Create a Knowledge Base first.")

        console.print(f"\n[cyan]Ingesting {len(documents)} documents into Knowledge Base...[/cyan]")
        self.upload_documents(documents)
        job_id = self.start_ingestion(kb_id or "demo-kb-id")
        return self.wait_for_ingestion(kb_id or "demo-kb-id", job_id)
