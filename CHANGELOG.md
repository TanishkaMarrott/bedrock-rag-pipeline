# Changelog

## [1.0.0] — 2026-04-30

### Added
- `BedrockRetriever` with dual-mode operation: `retrieve_and_generate` and `retrieve`
- DEMO_MODE for running the full pipeline without AWS credentials
- Pydantic v2 schema validation for `Document`, `RetrievedChunk`, `RAGResponse`
- `citations()` helper — deduplicates sources across retrieved chunks
- Knowledge Base ingestion pipeline with S3 sync and chunk configuration
- GitHub Actions CI — ruff lint + pytest in DEMO_MODE

### Changed
- `RetrievedChunk.source` replaces `source_doc` for a cleaner API surface
- `BedrockRetriever` client is now optional — DEMO_MODE works with zero config

### Fixed
- Schema field mismatches between `Document` and `RAGResponse` models
- `BedrockRetriever` required a live `BedrockClient` even in demo mode
