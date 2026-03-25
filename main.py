"""
Bedrock RAG Pipeline — Entry Point

Usage:
  python main.py                     # interactive query mode
  python main.py "your question"     # single query
  python demo/run_demo.py            # full demo with sample docs
"""

from __future__ import annotations

import sys
from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from rag.bedrock_client import BedrockClient
from rag.retriever import BedrockRetriever

console = Console()


def run_query(query: str) -> None:
    client = BedrockClient()
    retriever = BedrockRetriever(client)

    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}")
    response = retriever.retrieve_and_generate(query)

    console.print(Rule("Answer"))
    console.print(response.answer)

    if response.retrieved_chunks:
        console.print(Rule("Retrieved Chunks"))
        for i, chunk in enumerate(response.retrieved_chunks, 1):
            console.print(f"\n[dim]{i}. [{chunk.source_doc}] score={chunk.score:.2f}[/dim]")
            console.print(f"   {chunk.content[:200]}...")

    console.print(f"\n[dim]Sources: {', '.join(response.citations())}[/dim]")


def interactive_mode() -> None:
    console.print(Panel("[bold]Bedrock RAG Pipeline[/bold]\nType your question. Ctrl+C to exit.", expand=False))
    while True:
        try:
            query = input("\nQuery: ").strip()
            if query:
                run_query(query)
        except KeyboardInterrupt:
            console.print("\n[dim]Exiting.[/dim]")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_query(" ".join(sys.argv[1:]))
    else:
        interactive_mode()
