"""Content-hash helpers for indexing and Chroma deduplication."""
from __future__ import annotations

import hashlib
from typing import Any


def chunk_content_hash(page_content: str, source: str) -> str:
    return hashlib.md5(f"{page_content}{source}".encode("utf-8")).hexdigest()


def _inner_meta(meta: dict) -> dict:
    if not isinstance(meta, dict):
        return {}
    inner = meta.get("metadata", meta) if isinstance(meta.get("metadata"), dict) else meta
    return inner if isinstance(inner, dict) else meta


def meta_content_hashes(meta: dict) -> tuple[str | None, str | None]:
    inner = _inner_meta(meta)
    ch = inner.get("content_hash") or meta.get("content_hash")
    orig = inner.get("original_content_hash") or meta.get("original_content_hash")
    return (
        ch if isinstance(ch, str) else None,
        orig if isinstance(orig, str) else None,
    )


def existing_hash_sets_from_metadatas(metadatas: list) -> tuple[set[str], set[str]]:
    """Build (existing_hashes, original_existing_hashes) from a Chroma get() metadatas list."""
    existing_hashes: set[str] = set()
    original_existing_hashes: set[str] = set()
    for raw in metadatas:
        meta = raw if isinstance(raw, dict) else {}
        hv, orig = meta_content_hashes(meta)
        if hv:
            existing_hashes.add(hv)
            original_existing_hashes.add(hv)
        if orig:
            existing_hashes.add(orig)
            original_existing_hashes.add(orig)
    return existing_hashes, original_existing_hashes


def mark_hashes_and_split_fresh(
    texts: list[Any],
    initial_existing: set[str],
) -> tuple[list[Any], list[dict[str, Any]], int]:
    """Set content_hash on each document; return chunks not yet in initial_existing."""
    existing_hashes = set(initial_existing)
    new_texts: list[Any] = []
    new_hashes: list[dict[str, Any]] = []
    duplicate_count = 0
    for idx, text in enumerate(texts, 1):
        source = text.metadata.get("source", "unknown")
        ch = chunk_content_hash(text.page_content, source)
        text.metadata["content_hash"] = ch
        if ch in existing_hashes:
            duplicate_count += 1
        else:
            new_texts.append(text)
            new_hashes.append({"hash": ch, "source": source, "index": idx})
            existing_hashes.add(ch)
    return new_texts, new_hashes, duplicate_count


def strip_hashes_already_in_original_store(
    new_texts: list[Any],
    new_hashes: list[dict[str, Any]],
    original_existing_hashes: set[str],
) -> tuple[list[Any], list[dict[str, Any]], int]:
    """Remove pending chunks whose hash already exists in the store's original-hash set."""
    if not original_existing_hashes or not new_hashes:
        return new_texts, new_hashes, 0
    new_hash_set = {h["hash"] for h in new_hashes}
    intersection = original_existing_hashes.intersection(new_hash_set)
    if not intersection:
        return new_texts, new_hashes, 0
    filtered_texts: list[Any] = []
    filtered_hashes: list[dict[str, Any]] = []
    removed_count = 0
    for text, hash_info in zip(new_texts, new_hashes, strict=True):
        if hash_info["hash"] not in intersection:
            filtered_texts.append(text)
            filtered_hashes.append(hash_info)
        else:
            removed_count += 1
    return filtered_texts, filtered_hashes, removed_count


def duplicate_content_hash_extra_ids(ids: list[str], metadatas: list) -> list[str]:
    """Return ids to delete: all but the first document per content_hash."""
    hash_to_ids: dict[str, list[str]] = {}
    for i, doc_id in enumerate(ids):
        meta = metadatas[i] if i < len(metadatas) else {}
        if not isinstance(meta, dict):
            meta = {}
        inner = _inner_meta(meta)
        ch = inner.get("content_hash") if isinstance(inner, dict) else None
        if not isinstance(ch, str):
            ch = meta.get("content_hash") if isinstance(meta.get("content_hash"), str) else None
        if isinstance(ch, str):
            hash_to_ids.setdefault(ch, []).append(doc_id)
    out: list[str] = []
    for id_list in hash_to_ids.values():
        if len(id_list) > 1:
            out.extend(id_list[1:])
    return out
