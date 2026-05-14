"""Tests for dedupe helpers (no Chroma / Ollama)."""
import dedupe
from langchain_core.documents import Document


def test_existing_hash_sets_empty():
    a, b = dedupe.existing_hash_sets_from_metadatas([])
    assert a == set() and b == set()


def test_existing_hash_sets_nested_metadata():
    metas = [
        {"metadata": {"content_hash": "h1", "original_content_hash": "o1"}},
        {"content_hash": "h2"},
    ]
    existing, original = dedupe.existing_hash_sets_from_metadatas(metas)
    assert "h1" in existing and "h2" in existing
    assert "h1" in original and "o1" in existing
    assert "h2" in original


def test_existing_hash_sets_no_hash():
    metas = [{"metadata": {"source": "x.md"}}]
    existing, original = dedupe.existing_hash_sets_from_metadatas(metas)
    assert existing == set() and original == set()


def test_mark_hashes_and_split_fresh_skips_known():
    d1 = Document(page_content="hello", metadata={"source": "a.md"})
    d2 = Document(page_content="hello", metadata={"source": "a.md"})
    h = dedupe.chunk_content_hash("hello", "a.md")
    new_texts, new_hashes, dup = dedupe.mark_hashes_and_split_fresh([d1, d2], {h})
    assert dup == 2
    assert new_texts == [] and new_hashes == []


def test_mark_hashes_and_split_fresh_adds_new():
    d1 = Document(page_content="alpha", metadata={"source": "a.md"})
    new_texts, new_hashes, dup = dedupe.mark_hashes_and_split_fresh([d1], set())
    assert dup == 0
    assert len(new_texts) == 1
    assert new_hashes[0]["hash"] == d1.metadata["content_hash"]


def test_strip_hashes_already_in_original_store():
    d = Document(page_content="x", metadata={"source": "s.md"})
    h = dedupe.chunk_content_hash("x", "s.md")
    d.metadata["content_hash"] = h
    new_hashes = [{"hash": h, "source": "s.md", "index": 1}]
    ft, fh, removed = dedupe.strip_hashes_already_in_original_store([d], new_hashes, {h})
    assert removed == 1
    assert ft == [] and fh == []


def test_duplicate_content_hash_extra_ids():
    ids = ["a", "b", "c", "d"]
    metas = [
        {"content_hash": "same"},
        {"content_hash": "same"},
        {"content_hash": "same"},
        {"content_hash": "other"},
    ]
    extra = dedupe.duplicate_content_hash_extra_ids(ids, metas)
    assert set(extra) == {"b", "c"}


def test_duplicate_content_hash_inner_meta():
    ids = ["x", "y"]
    metas = [
        {"metadata": {"content_hash": "dup"}},
        {"metadata": {"content_hash": "dup"}},
    ]
    extra = dedupe.duplicate_content_hash_extra_ids(ids, metas)
    assert extra == ["y"]
