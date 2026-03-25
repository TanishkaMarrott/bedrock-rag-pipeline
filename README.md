# bedrock-rag-pipeline

RAG pipeline using AWS Bedrock Knowledge Base and Claude — document ingestion, semantic retrieval, and grounded generation.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-orange)
![Claude](https://img.shields.io/badge/LLM-Claude%203.5%20Sonnet-purple)
![RAG](https://img.shields.io/badge/Pattern-RAG-green)
![Demo mode](https://img.shields.io/badge/Demo-no%20credentials%20needed-brightgreen)

---

## What This Is

A RAG pipeline built on AWS Bedrock's native Knowledge Base service. Bedrock handles the full vector pipeline — chunking, embedding, and indexing — so you can focus on ingestion and query logic.

---

## Architecture

```
Documents (S3)
      │
      ▼
┌─────────────────────────────────────────┐
│        Bedrock Knowledge Base           │
│                                         │
│  Chunk → Embed (Titan) → Index          │
│          (OpenSearch Serverless)        │
└──────────────────┬──────────────────────┘
                   │
         RetrieveAndGenerate API
                   │
         ┌─────────┴──────────┐
         │  Retrieve top-k    │
         │  relevant chunks   │
         └─────────┬──────────┘
                   │ chunks + query
                   ▼
            Claude 3.5 Sonnet
            (grounded answer
             with citations)
```

### Why Bedrock Knowledge Base Over DIY RAG

| | DIY RAG (Qdrant + LangChain) | Bedrock Knowledge Base |
|---|---|---|
| **Setup** | Provision vector DB, configure embeddings | Point at S3 bucket |
| **Chunking** | Write chunking logic | Configurable strategies built-in |
| **Embeddings** | Manage embedding model | Amazon Titan handles it |
| **Scaling** | Manage infrastructure | Fully managed |
| **Citations** | Manual | Built-in source attribution |

Bedrock KB is the production choice for AWS-native teams. This pipeline shows how to use it.

---

## Key APIs

| API | Client | What it does |
|---|---|---|
| `RetrieveAndGenerate` | `bedrock-agent-runtime` | Full RAG in one call — retrieve + generate |
| `Retrieve` | `bedrock-agent-runtime` | Retrieval only — you control generation |
| `StartIngestionJob` | `bedrock-agent` | Trigger document ingestion into KB |
| `InvokeModel` | `bedrock-runtime` | Direct Claude invocation |

---

## Quick Start

```bash
git clone https://github.com/TanishkaMarrott/bedrock-rag-pipeline.git
cd bedrock-rag-pipeline
pip install -r requirements.txt
cp .env.example .env

# Demo mode — no AWS credentials needed
DEMO_MODE=true python demo/run_demo.py

# Interactive query mode
DEMO_MODE=true python main.py

# Real AWS — add credentials + KB ID to .env
DEMO_MODE=false python main.py "What is Bedrock Knowledge Base?"
```

---

## Project Structure

```
bedrock-rag-pipeline/
├── rag/
│   ├── bedrock_client.py    # boto3 client wrapper (3 clients)
│   ├── knowledge_base.py    # S3 upload + ingestion job management
│   └── retriever.py         # RetrieveAndGenerate + Retrieve
├── models/
│   └── schemas.py           # Document, RAGResponse, RetrievedChunk
├── demo/
│   ├── run_demo.py          # End-to-end demo with sample docs
│   └── sample_docs/         # AWS Bedrock documentation samples
└── main.py                  # Interactive query mode
```

---

## Related

- [aws-sar-mcp](https://github.com/TanishkaMarrott/aws-sar-mcp) — MCP server for AWS IAM actions
- [mem0-pipeline](https://github.com/TanishkaMarrott/mem0-pipeline) — Alternative memory pipeline using mem0 + Qdrant + Neo4j

---

## Author

Built by [Tanishka Marrott](https://github.com/TanishkaMarrott) — AI Agent Systems Engineer
