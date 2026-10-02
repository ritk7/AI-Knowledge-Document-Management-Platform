# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Documentation

- Added Python version, backend, and local-generation badges under the
  README title for an at-a-glance summary of the stack.
- Added runnable `curl` examples for `POST /documents/upload` and
  `POST /query` to the API reference (previously only JSON schemas were
  shown, with no example of actually calling the endpoints).
- Added Linux and Windows install commands for Ollama in the Setup section
  (previously only `brew install ollama` for macOS was shown).
- Documented example responses for `GET /config` and `GET /health` in the
  API reference, completing coverage for all six endpoints.
- Documented example responses for `/documents/upload`, `GET /documents`,
  and `DELETE /documents/{id}` in the API reference (previously only
  `/query` had one).
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
