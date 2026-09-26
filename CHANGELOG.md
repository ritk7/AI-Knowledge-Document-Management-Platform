# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Documentation

- Added a "Requirements at a glance" summary (disk space, downloads, Python
  version) at the top of the Setup section.
- Added a Troubleshooting section covering common setup issues.
- Documented previously-missing storage config variables in the
  Configuration table.
- Added the `docs/` browser demo to the project layout tree.
- Fixed an incorrect directory name in the setup instructions.

## Retrieval pipeline

- Fixed the pipeline hanging permanently at the re-rank stage.
- Drew document territories on the embedding map.
- Added a browser-native demo of the retrieval pipeline, served from
  GitHub Pages.

## Generation & robustness

- Switched generation to local Ollama; fixed upload/retrieval bugs found
  by adversarial testing.

## Initial release

- Built the hybrid RAG document Q&A platform with a measured retrieval
  pipeline: page-aware chunking, hybrid BM25 + embedding search,
  cross-encoder re-ranking, citations, and low-confidence abstention.
