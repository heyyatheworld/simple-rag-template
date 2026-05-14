# Simple RAG Template

A minimal RAG (Retrieval-Augmented Generation) pipeline that indexes Markdown documents and answers questions using **Ollama** (local LLM + embeddings) and **ChromaDB** (vector store). No API keys required.

## Requirements

- **Python 3.9+**
- **Ollama** installed and running ([ollama.com](https://ollama.com))

Pull the models used by default:

```bash
ollama pull mxbai-embed-large:latest
ollama pull llama3.2
```

## Install

```bash
git clone <repo-url>
cd simple-rag-template
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Copy the example env and adjust if needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBEDDING_MODEL` | `mxbai-embed-large:latest` | Embedding model |
| `OLLAMA_LLM_MODEL` | `llama3.2` | Chat model for answers |
| `OLLAMA_CHROMA_PERSIST_DIR` | (empty → `vectors/chroma_db`) | ChromaDB data directory |
| `OLLAMA_CHROMA_COLLECTION` | `rag_collection` | Collection name |
| `OLLAMA_VECTOR_SIZE` | `1024` | Embedding dimension |
| `OLLAMA_DOCS_PATH` | `data` | Directory with `.md` files (incl. subdirs) |
| `OLLAMA_VECTORS_PATH` | `vectors` | Base path for vector DB |
| `OLLAMA_EMBED_MAX_DOC_CHARS` | `500` | Max chars per chunk sent to embed (long texts are split) |

## Usage

**Index** all `.md` files under `data/` (and subdirs) into the vector DB:

```bash
python rag_cli.py --index
```

**Ask a question** (uses retrieved context + LLM):

```bash
python rag_cli.py --query "What is argparse?"
```

**Status** — read-only: collection, doc count, duplicate report (no writes), and config:

```bash
python rag_cli.py --status
```

**Dedupe** — remove extra chunks that share the same `content_hash` (keeps one per hash):

```bash
python rag_cli.py --dedupe
```

**Clear** the vector collection and start over:

```bash
python rag_cli.py --clear
```

## Project layout

```
.
├── rag_cli.py       # CLI entry (--index, --query, --status, --dedupe, --clear)
├── rag_pipeline.py  # RAG pipeline: load docs, split, embed, ChromaDB, Ollama LLM
├── requirements.txt
├── .env.example
├── .env             # Your config (not committed)
├── data/            # Markdown docs to index (default; recursive)
└── vectors/         # ChromaDB data (default: vectors/chroma_db)
```

## How it works

1. **Index**: Reads all `.md` files from `data/` (recursive), splits by Markdown headers and by size (chunk size 1000, overlap 200). Deduplicates by content hash, then embeds via Ollama and stores in ChromaDB. Long chunks are trimmed to `OLLAMA_EMBED_MAX_DOC_CHARS` per embed request.
2. **Query**: Embeds the question, retrieves top 5 chunks from ChromaDB, and asks the Ollama LLM to answer from that context.
3. **Status**: Reports collection stats, sample sources, duplicate check (read-only), docs directory, and Ollama/retriever config. Use **Dedupe** to actually remove duplicate chunks.

## License

MIT (or your choice).
