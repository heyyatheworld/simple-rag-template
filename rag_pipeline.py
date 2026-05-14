"""RAG pipeline: index Markdown docs into ChromaDB, answer questions via Ollama."""
import hashlib
import os
import textwrap

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

import dedupe
import rag_config
import rag_logging

logger = rag_logging.get_logger()


class RAGPipeline:
    """Index Markdown docs into ChromaDB and answer questions using Ollama embeddings + LLM."""

    def __init__(self):
        load_dotenv()
        logger.info("")
        logger.info("[INIT] Initializing RAG Pipeline...")
        _root = os.path.dirname(os.path.abspath(__file__))
        self.cfg = rag_config.load_rag_config(_root)
        c = self.cfg
        self.collection_name = c.collection_name
        self.ollama_base_url = c.ollama_base_url
        self.embedding_model = c.embedding_model
        self.llm_model = c.llm_model
        self.vector_size = c.vector_size
        self.index_path = os.path.join(_root, c.docs_path)
        self.vectors = os.path.join(_root, c.vectors_path)
        self.chroma_persist_directory = c.chroma_persist_dir
        logger.info(f"[INIT] Docs path: {self.index_path}")
        logger.info(f"[INIT] ChromaDB path: {self.chroma_persist_directory}")
        logger.info("")

        logger.info(f"[INIT] Creating Ollama Embeddings ({self.embedding_model})...")
        self.embeddings = OllamaEmbeddings(
            model=self.embedding_model,
            base_url=self.ollama_base_url,
        )

        logger.info(f"[INIT] ChromaDB collection: '{self.collection_name}'")
        os.makedirs(self.chroma_persist_directory, exist_ok=True)
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.chroma_persist_directory,
            embedding_function=self.embeddings,
        )

        logger.info(f"[INIT] Retriever (k={c.retriever_k})")
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": c.retriever_k})

        logger.info("[INIT] PromptTemplate (from env / prompts/answer.txt)")
        self.prompt = PromptTemplate.from_template(c.prompt_template)

        logger.info(f"[INIT] LLM Ollama ({self.llm_model}, temperature={c.llm_temperature})")
        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=c.llm_temperature,
            base_url=self.ollama_base_url,
        )

        def format_docs(docs):
            parts = []
            for doc in docs:
                src = doc.metadata.get("source", "unknown")
                parts.append(f"[Source: {src}]\n{doc.page_content}")
            return "\n\n".join(parts)

        self._format_docs = format_docs
        self._qa_chain = self.prompt | self.llm | StrOutputParser()
        logger.info("[INIT] QA chain ready (retrieve + prompt + LLM)")
        logger.info("[INIT] Done\n")

    def index(self):
        """Load .md from docs path, split, deduplicate, embed via Ollama, store in ChromaDB."""
        logger.info("[INDEX] Starting...")
        logger.info(f"[INDEX] Reading from {self.index_path} (incl. subdirs)")

        documents = []
        md_files = []
        for root, _dirs, files in os.walk(self.index_path):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.index_path)
                    md_files.append((file_path, rel_path))
        logger.info(f"[INDEX] Found {len(md_files)} .md files")

        for file_path, rel_path in md_files:
            logger.debug("Reading %s", rel_path)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                doc = Document(page_content=content, metadata={"source": rel_path})
                documents.append(doc)
        logger.info(f"[INDEX] Loaded {len(documents)} documents")

        logger.info("[INDEX] Splitting by markdown headers...")

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),])

        split_docs = []
        for doc in documents:
            header_splits = markdown_splitter.split_text(doc.page_content)
            for split in header_splits:
                split.metadata.update(doc.metadata)
            split_docs.extend(header_splits)

        logger.info(f"[INDEX] After header split: {len(split_docs)} sections")

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=self.cfg.tiktoken_encoding,
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
            separators=["\n\n## ", "\n\n", "\n", ". ", " ", ""])
        texts = text_splitter.split_documents(split_docs)
        logger.info(f"[INDEX] Chunks: {len(texts)}")

        logger.info("[INDEX] Checking duplicates...")

        existing_hashes: set[str] = set()
        original_existing_hashes: set[str] = set()
        try:
            result = self.vector_store._collection.get(
                include=["metadatas"],
                limit=100000,
            )
            metadatas = result.get("metadatas") or []
            existing_hashes, original_existing_hashes = dedupe.existing_hash_sets_from_metadatas(metadatas)
            logger.info(f"[INDEX] Existing documents: {len(existing_hashes)}")
        except Exception as e:
            logger.warning(f"[INDEX] Error loading existing: {e}")
            existing_hashes = set()
            original_existing_hashes = set()

        new_texts, new_hashes, duplicate_count = dedupe.mark_hashes_and_split_fresh(texts, existing_hashes)
        ft, fh, removed = dedupe.strip_hashes_already_in_original_store(
            new_texts, new_hashes, original_existing_hashes
        )
        new_texts, new_hashes = ft, fh
        duplicate_count += removed

        logger.info(f"[INDEX] Duplicates skipped: {duplicate_count}, new to add: {len(new_texts)}")
        if not new_texts:
            logger.info("[INDEX] All documents already in collection. Skip.")
            return

        logger.info("[INDEX] Adding to vector store...")
        max_chars_per_doc = self.cfg.embed_max_doc_chars

        def trim_doc(d):
            if len(d.page_content) <= max_chars_per_doc:
                return [d]
            out = []
            start = 0
            idx = 0
            base_hash = d.metadata.get("content_hash", "")
            while start < len(d.page_content):
                end = start + max_chars_per_doc
                chunk_content = d.page_content[start:end]
                meta = dict(d.metadata)
                meta["_chunk_index"] = idx
                meta["original_content_hash"] = base_hash
                meta["content_hash"] = hashlib.md5(
                    (base_hash + str(idx)).encode()
                ).hexdigest()
                out.append(Document(page_content=chunk_content, metadata=meta))
                start = end
                idx += 1
            return out

        expanded = []
        for doc in new_texts:
            expanded.extend(trim_doc(doc))
        if len(expanded) != len(new_texts):
            logger.info(f"[INDEX] {len(new_texts)} chunks → {len(expanded)} docs (trim to {max_chars_per_doc} chars each)")

        hashes_written = {doc.metadata.get("content_hash") for doc in expanded if doc.metadata.get("content_hash")}

        total = len(expanded)
        batch_size = self.cfg.index_batch_size
        num_batches = (total + batch_size - 1) // batch_size if total else 0
        log_interval = 1 if num_batches <= 40 else max(1, (num_batches + 19) // 20)
        logger.info(f"[INDEX] Adding in batches of {batch_size} ({num_batches} batch(es))...")
        for bi, start in enumerate(range(0, total, batch_size)):
            end = min(start + batch_size, total)
            self.vector_store.add_documents(expanded[start:end])
            if (bi + 1) % log_interval == 0 or end == total:
                logger.info(f"[INDEX] Added {end}/{total} documents")
        logger.info(f"[INDEX] Added {total} documents total")

        logger.info("[INDEX] Verifying...")
        try:
            db_hashes = set()
            page_size = 10_000
            offset = 0
            while True:
                result = self.vector_store._collection.get(
                    include=["metadatas"],
                    limit=page_size,
                    offset=offset,
                )
                metadatas = result.get("metadatas") or []
                ids = result.get("ids") or []
                if not ids:
                    break
                for meta in metadatas:
                    if not isinstance(meta, dict):
                        continue
                    h = meta.get("content_hash") or (meta.get("metadata") or {}).get("content_hash")
                    if h:
                        db_hashes.add(h)
                if len(ids) < page_size:
                    break
                offset += page_size
            found = len(hashes_written & db_hashes)
            missing = len(hashes_written - db_hashes)
            logger.info(f"[INDEX] Hashes in DB: {found}/{len(hashes_written)}")
            if missing:
                logger.warning(f"[INDEX] Missing: {missing}")
        except Exception as e:
            logger.warning(f"[INDEX] Verify error: {e}")

        logger.info("[INDEX] Done\n")

    def answer(self, query: str) -> str:
        """Run RAG: retrieve context from ChromaDB, then generate answer with LLM."""
        logger.info(f"[ANSWER] Query: {query[:80]}...")
        docs = self.retriever.invoke(query)
        context = self._format_docs(docs)
        response = self._qa_chain.invoke({"context": context, "question": query})
        logger.info(f"[ANSWER] Response ({len(response)} chars)")
        for line in textwrap.wrap(response, 80):
            logger.info(f"[ANSWER] {line}")
        sources = sorted({doc.metadata.get("source", "?") for doc in docs})
        if sources:
            logger.info("[ANSWER] Sources:")
            for s in sources:
                logger.info(f"  - {s}")
        logger.info("")
        return response

    def status(self):
        """Print vector DB stats, sample docs, duplicate check, and config."""
        logger.info("=" * 80)
        logger.info("VECTOR DB STATUS (ChromaDB)")
        logger.info("=" * 80)

        if not hasattr(self, "vector_store") or self.vector_store is None:
            logger.error("Error: ChromaDB not initialized")
            return

        logger.info("\n[STATUS] 1. CHROMADB")
        logger.info("-" * 80)
        try:
            logger.info(f"  Path: {self.chroma_persist_directory}")
            logger.info(f"  Collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"  Error: {e}")
            return

        points_count = 0
        logger.info(f"\n[STATUS] 2. COLLECTION '{self.collection_name}'")
        logger.info("-" * 80)
        try:
            points_count = self.vector_store._collection.count()
            logger.info(f"  Vector size: {self.vector_size}")
            logger.info(f"  Documents: {points_count}")
        except Exception as e:
            logger.warning(f"  Error: {e}")
            return

        logger.info("\n[STATUS] 3. SAMPLE DOCUMENTS")
        logger.info("-" * 80)
        try:
            result = self.vector_store._collection.get(
                limit=10,
                include=["metadatas"],
            )
            ids = result.get("ids") or []
            metadatas = result.get("metadatas") or []
            logger.info(f"  Total in collection: {points_count}")

            if ids:
                sources = {}
                for i in range(len(ids)):
                    meta = metadatas[i] if i < len(metadatas) else {}
                    if not isinstance(meta, dict):
                        meta = {}
                    inner = meta.get("metadata", meta)
                    source = inner.get("source", meta.get("source", "unknown")) if isinstance(inner, dict) else meta.get("source", "unknown")
                    sources[source] = sources.get(source, 0) + 1
                logger.info("  By source:")
                for source, cnt in list(sources.items())[:10]:
                    logger.info(f"    - {source}: {cnt} chunks")
            else:
                logger.info("  Collection empty")
        except Exception as e:
            logger.warning(f"  Error: {e}")

        logger.info("\n[STATUS] 4. DUPLICATES CHECK")
        logger.info("-" * 80)
        try:
            if points_count == 0:
                logger.info("  (empty, skip)")
            else:
                result = self.vector_store._collection.get(
                    include=["metadatas"],
                    limit=100000,
                )
                all_ids = result.get("ids") or []
                all_metadatas = result.get("metadatas") or []
                hash_to_points = {}
                points_without_hash = []
                for i, doc_id in enumerate(all_ids):
                    meta = all_metadatas[i] if i < len(all_metadatas) else {}
                    if not isinstance(meta, dict):
                        meta = {}
                    inner = meta.get("metadata", meta)
                    content_hash = (inner.get("content_hash") if isinstance(inner, dict) else None) or meta.get("content_hash")
                    if content_hash:
                        hash_to_points.setdefault(content_hash, []).append({"id": doc_id, "meta": meta})
                    else:
                        points_without_hash.append(doc_id)
                duplicates = {h: lst for h, lst in hash_to_points.items() if len(lst) > 1}
                logger.info(f"  With hash: {len(hash_to_points)}, without: {len(points_without_hash)}, duplicate groups: {len(duplicates)}")
                if duplicates:
                    dup_chunks = sum(len(lst) - 1 for lst in duplicates.values())
                    logger.info(f"  Duplicate chunks (beyond first per hash): {dup_chunks}")
                    logger.info("  (read-only; run --dedupe to remove extras)")
                else:
                    logger.info("  No duplicates")
        except Exception as e:
            logger.warning(f"  Error: {e}")

        logger.info("\n[STATUS] 5. DOCS DIRECTORY (incl. subdirs)")
        logger.info("-" * 80)
        try:
            if os.path.exists(self.index_path):
                md_files = []
                for root, _dirs, files in os.walk(self.index_path):
                    for file in files:
                        if file.endswith(".md"):
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, self.index_path)
                            md_files.append((rel_path, file_path))
                logger.info(f"  Path: {self.index_path}")
                logger.info(f"  .md files: {len(md_files)}")
                total_size = sum(os.path.getsize(fp) for _, fp in md_files)
                for rel_path, file_path in sorted(md_files)[:15]:
                    size = os.path.getsize(file_path)
                    logger.info(f"    {rel_path}: {size:,} B")
                if len(md_files) > 15:
                    logger.info(f"    ... and {len(md_files) - 15} more")
                logger.info(f"  Total: {total_size:,} B ({total_size/1024:.1f} KB)")
            else:
                logger.info(f"  Path missing: {self.index_path}")
        except Exception as e:
            logger.warning(f"  Error: {e}")

        logger.info("\n[STATUS] 6. EMBEDDINGS (Ollama)")
        logger.info("-" * 80)
        logger.info(f"  Model: {self.embedding_model}, dim: {self.vector_size}")
        logger.info(f"  URL: {self.ollama_base_url}")

        logger.info("\n[STATUS] 7. LLM (Ollama)")
        logger.info("-" * 80)
        logger.info(f"  Model: {self.llm_model}, temperature: {self.cfg.llm_temperature}")

        logger.info("\n[STATUS] 7b. PROMPT & CHUNKING")
        logger.info("-" * 80)
        logger.info(f"  Chunk size / overlap (tokens): {self.cfg.chunk_size} / {self.cfg.chunk_overlap}")
        logger.info(f"  Tiktoken encoding: {self.cfg.tiktoken_encoding}")
        logger.info(f"  Embed max chars: {self.cfg.embed_max_doc_chars}, index batch: {self.cfg.index_batch_size}")

        logger.info("\n[STATUS] 8. RETRIEVER")
        logger.info("-" * 80)
        logger.info(f"  k: {self.retriever.search_kwargs.get('k', 'N/A')}")

        logger.info("\n[STATUS] 9. TEST SEARCH")
        logger.info("-" * 80)
        try:
            if points_count > 0:
                results = self.retriever.invoke("Python")
                logger.info(f"  Query 'Python': {len(results)} results")
            else:
                logger.info("  (empty)")
        except Exception as e:
            logger.warning(f"  Error: {e}")

        logger.info("\n" + "=" * 80)
        logger.info("STATUS DONE")
        logger.info("=" * 80 + "\n")

    def dedupe(self):
        """Remove duplicate chunks in Chroma (same content_hash), keeping one per hash."""
        logger.info("[DEDUPE] Scanning collection for duplicate content_hash...")
        if not hasattr(self, "vector_store") or self.vector_store is None:
            logger.error("[DEDUPE] Error: ChromaDB not initialized")
            return
        try:
            points_count = self.vector_store._collection.count()
        except Exception as e:
            logger.warning(f"[DEDUPE] Error: {e}")
            return
        if points_count == 0:
            logger.info("[DEDUPE] Collection empty, nothing to do.")
            return
        try:
            result = self.vector_store._collection.get(
                include=["metadatas"],
                limit=100000,
            )
            all_ids = result.get("ids") or []
            all_metadatas = result.get("metadatas") or []
            ids_to_delete = dedupe.duplicate_content_hash_extra_ids(all_ids, all_metadatas)
            if not ids_to_delete:
                logger.info("[DEDUPE] No duplicates found.")
                return
            batch_size = 100
            deleted_count = 0
            for i in range(0, len(ids_to_delete), batch_size):
                batch_ids = ids_to_delete[i : i + batch_size]
                try:
                    self.vector_store.delete(ids=batch_ids)
                    deleted_count += len(batch_ids)
                except Exception as e:
                    logger.warning(f"[DEDUPE] Delete error: {e}")
            logger.info(f"[DEDUPE] Removed {deleted_count} duplicate chunk(s), kept one per hash.")
        except Exception as e:
            logger.warning(f"[DEDUPE] Error: {e}")

    def clear(self):
        """Delete the ChromaDB collection and recreate it empty."""
        logger.info("[CLEAR] Clearing ChromaDB collection...")
        try:
            self.vector_store._client.delete_collection(name=self.collection_name)
        except Exception as e:
            logger.warning(f"[CLEAR] Warning: {e}")
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.chroma_persist_directory,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.cfg.retriever_k})
        logger.info("[CLEAR] Done")

    def close(self):
        """No-op; ChromaDB persists to disk."""
        pass
