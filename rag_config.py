"""Load RAG settings from environment (local template defaults)."""
from __future__ import annotations

import os
from dataclasses import dataclass

_DEFAULT_PROMPT = """You answer using only the context below. Do not invent facts that are not supported by the context. If the context is insufficient, say so briefly. When you state a fact from the context, name the source file shown in the [Source: …] line for that passage.

Context:
{context}

Question:
{question}

Answer:
"""


def _i(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def _f(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def _s(key: str, default: str) -> str:
    raw = os.getenv(key)
    return default if raw is None else raw


def _load_prompt_template(package_dir: str) -> str:
    custom = os.getenv("OLLAMA_PROMPT_PATH", "").strip()
    if custom:
        path = custom if os.path.isabs(custom) else os.path.join(package_dir, custom)
    else:
        path = os.path.join(package_dir, "prompts", "answer.txt")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            return text
    return _DEFAULT_PROMPT


@dataclass(frozen=True)
class RAGConfig:
    collection_name: str
    ollama_base_url: str
    embedding_model: str
    llm_model: str
    vector_size: int
    docs_path: str
    vectors_path: str
    chroma_persist_dir: str
    retriever_k: int
    llm_temperature: float
    chunk_size: int
    chunk_overlap: int
    tiktoken_encoding: str
    embed_max_doc_chars: int
    index_batch_size: int
    prompt_template: str


def load_rag_config(package_dir: str) -> RAGConfig:
    """Read all OLLAMA_* settings used by the pipeline (after dotenv load)."""
    vectors_path = _s("OLLAMA_VECTORS_PATH", "vectors")
    chroma_persist = os.getenv("OLLAMA_CHROMA_PERSIST_DIR", "").strip() or os.path.join(
        package_dir, vectors_path, "chroma_db"
    )
    chunk_size = max(1, _i("OLLAMA_CHUNK_SIZE", 1000))
    chunk_overlap = max(0, _i("OLLAMA_CHUNK_OVERLAP", 200))
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size - 1)
    return RAGConfig(
        collection_name=_s("OLLAMA_CHROMA_COLLECTION", "rag_collection"),
        ollama_base_url=_s("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        embedding_model=_s("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large:latest"),
        llm_model=_s("OLLAMA_LLM_MODEL", "llama3.2"),
        vector_size=_i("OLLAMA_VECTOR_SIZE", 1024),
        docs_path=_s("OLLAMA_DOCS_PATH", "data"),
        vectors_path=vectors_path,
        chroma_persist_dir=chroma_persist,
        retriever_k=max(1, _i("OLLAMA_RETRIEVER_K", 5)),
        llm_temperature=_f("OLLAMA_LLM_TEMPERATURE", 0.0),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tiktoken_encoding=_s("OLLAMA_TIKTOKEN_ENCODING", "cl100k_base"),
        embed_max_doc_chars=max(1, _i("OLLAMA_EMBED_MAX_DOC_CHARS", 500)),
        index_batch_size=max(1, _i("OLLAMA_INDEX_BATCH_SIZE", 32)),
        prompt_template=_load_prompt_template(package_dir),
    )
