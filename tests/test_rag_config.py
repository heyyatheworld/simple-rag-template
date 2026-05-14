"""Tests for rag_config (no Chroma / Ollama)."""
import textwrap

import rag_config


def test_retriever_k_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_RETRIEVER_K", "8")
    cfg = rag_config.load_rag_config(str(tmp_path))
    assert cfg.retriever_k == 8


def test_chunk_overlap_clamped_when_too_large(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_CHUNK_SIZE", "100")
    monkeypatch.setenv("OLLAMA_CHUNK_OVERLAP", "150")
    cfg = rag_config.load_rag_config(str(tmp_path))
    assert cfg.chunk_size == 100
    assert cfg.chunk_overlap < cfg.chunk_size


def test_custom_prompt_path_absolute(monkeypatch, tmp_path):
    custom = tmp_path / "my.txt"
    custom.write_text(
        textwrap.dedent(
            """\
            C: {context}
            Q: {question}
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OLLAMA_PROMPT_PATH", str(custom))
    cfg = rag_config.load_rag_config(str(tmp_path))
    assert "{context}" in cfg.prompt_template
    assert "C:" in cfg.prompt_template


def test_default_prompt_has_placeholders(monkeypatch, tmp_path):
    monkeypatch.delenv("OLLAMA_PROMPT_PATH", raising=False)
    cfg = rag_config.load_rag_config(str(tmp_path))
    assert "{context}" in cfg.prompt_template
    assert "{question}" in cfg.prompt_template
