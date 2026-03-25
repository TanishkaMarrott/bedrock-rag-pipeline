"""
Demo runner — loads sample documents and runs 3 test queries.
Set DEMO_MODE=true (default) to run without AWS credentials.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from models.schemas import Document
from rag.bedrock_client import BedrockClient
from rag.knowledge_base import KnowledgeBaseManager
from rag.retriever import BedrockRetriever

console = Console()

SAMPLE_QUERIES = [
    "What is AWS Bedrock and what models does it support?",
    "How does Bedrock Knowledge Base automate the RAG pipeline?",
    "What chunking strategies does Bedrock support?",
]

SAMPLE_DOCS = [
    Document(
        doc_id="bedrock-overview",
        title="AWS Bedrock Overview",
        content=(Path(__file__).parent / "sample_docs" / "bedrock-overview.txt").read_text(),
        source="bedrock-overview.txt",
    ),
    Document(
        doc_id="bedrock-kb-guide",
        title="Bedrock Knowledge Bases Guide",
        content=(Path(__file__).parent / "sample_docs" / "bedrock-kb-guide.txt").read_text(),
        source="bedrock-kb-guide.txt",
    ),
]


def main() -> None:
    console.print(Panel("[bold]Bedrock RAG Pipeline Demo[/bold]", expand=False))

    client = BedrockClient()
    kb_manager = KnowledgeBaseManager(client)
    retriever = BedrockRetriever(client)

    # Ingest documents
    kb_manager.ingest(SAMPLE_DOCS)

    # Run queries
    for query in SAMPLE_QUERIES:
        console.print(f"\n[bold cyan]Q:[/bold cyan] {query}")
        response = retriever.retrieve_and_generate(query)
        console.print(f"[bold green]A:[/bold green] {response.answer}")
        console.print(f"[dim]Sources: {', '.join(response.citations())}[/dim]")


if __name__ == "__main__":
    main()
