"""RAG pipeline: index Markdown docs into ChromaDB, answer questions via Ollama."""
import os
import time
import textwrap
import hashlib

from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma


class RAGPipeline:
    """Index Markdown docs into ChromaDB and answer questions using Ollama embeddings + LLM."""

    def __init__(self):
        print()
        print("[INIT] Initializing RAG Pipeline...")
        self.collection_name = os.getenv("OLLAMA_CHROMA_COLLECTION", "rag_collection")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large:latest")
        self.llm_model = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
        self.vector_size = int(os.getenv("OLLAMA_VECTOR_SIZE", "1024"))
        docs_path = os.getenv("OLLAMA_DOCS_PATH", "data")
        vectors_path = os.getenv("OLLAMA_VECTORS_PATH", "vectors")
        chroma_persist = os.getenv("OLLAMA_CHROMA_PERSIST_DIR", "").strip() or os.path.join(
            os.path.dirname(__file__), vectors_path, "chroma_db"
        )

        self.index_path = os.path.join(os.path.dirname(__file__), docs_path)
        self.vectors = os.path.join(os.path.dirname(__file__), vectors_path)
        self.chroma_persist_directory = chroma_persist
        print(f"[INIT] Docs path: {self.index_path}")
        print(f"[INIT] ChromaDB path: {self.chroma_persist_directory}")
        print()

        print(f"[INIT] Creating Ollama Embeddings ({self.embedding_model})...")
        self.embeddings = OllamaEmbeddings(
            model=self.embedding_model,
            base_url=self.ollama_base_url,
        )

        print(f"[INIT] ChromaDB collection: '{self.collection_name}'")
        os.makedirs(self.chroma_persist_directory, exist_ok=True)
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.chroma_persist_directory,
            embedding_function=self.embeddings,
        )

        print("[INIT] Retriever (k=5)")
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        print("[INIT] PromptTemplate")
        self.prompt = PromptTemplate.from_template(
            "Answer the following question based on the context:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        )

        print(f"[INIT] LLM Ollama ({self.llm_model})")
        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=0,
            base_url=self.ollama_base_url,
        )

        def format_docs(docs):
            formatted = "\n\n".join(doc.page_content for doc in docs)
            return formatted

        self._format_docs = format_docs
        print("[INIT] RAG chain ready")
        self.chain = (
            {"context": self.retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        print("[INIT] Done\n")

    def index(self):
        """Load .md from docs path, split, deduplicate, embed via Ollama, store in ChromaDB."""
        print("[INDEX] Starting...")
        print(f"[INDEX] Reading from {self.index_path} (incl. subdirs)")
        
        documents = []
        md_files = []
        for root, _dirs, files in os.walk(self.index_path):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.index_path)
                    md_files.append((file_path, rel_path))
        print(f"[INDEX] Found {len(md_files)} .md files")
        
        for file_path, rel_path in md_files:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                doc = Document(page_content=content, metadata={"source": rel_path})
                documents.append(doc)
        print(f"[INDEX] Loaded {len(documents)} documents")

        print("[INDEX] Splitting by markdown headers...")

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
        
        print(f"[INDEX] After header split: {len(split_docs)} sections")
        
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n## ", "\n\n", "\n", ". ", " ", ""])
        texts = text_splitter.split_documents(split_docs)
        print(f"[INDEX] Chunks: {len(texts)}")

        print("[INDEX] Checking duplicates...")

        existing_hashes = set()
        original_existing_hashes = set()
        try:
            result = self.vector_store._collection.get(
                include=["metadatas"],
                limit=100000,
            )
            ids = result.get("ids") or []
            metadatas = result.get("metadatas") or []
            for i, doc_id in enumerate(ids):
                meta = metadatas[i] if i < len(metadatas) else {}
                if not isinstance(meta, dict):
                    meta = {}
                inner = meta.get("metadata", meta) if isinstance(meta.get("metadata"), dict) else meta
                if not isinstance(inner, dict):
                    inner = meta
                hash_value = inner.get("content_hash") or meta.get("content_hash")
                orig = inner.get("original_content_hash") or meta.get("original_content_hash")
                if hash_value:
                    existing_hashes.add(hash_value)
                    original_existing_hashes.add(hash_value)
                if orig:
                    existing_hashes.add(orig)
                    original_existing_hashes.add(orig)
            print(f"[INDEX] Existing documents: {len(existing_hashes)}")
        except Exception as e:
            print(f"[INDEX] Error loading existing: {e}")
            existing_hashes = set()
            original_existing_hashes = set()
        
        new_texts = []
        duplicate_count = 0
        new_hashes = []
        for idx, text in enumerate(texts, 1):
            source = text.metadata.get("source", "unknown")
            content_hash = hashlib.md5(f"{text.page_content}{source}".encode("utf-8")).hexdigest()
            text.metadata["content_hash"] = content_hash
            if content_hash in existing_hashes:
                duplicate_count += 1
            else:
                new_texts.append(text)
                new_hashes.append({"hash": content_hash, "source": source, "index": idx})
                existing_hashes.add(content_hash)
        
        print(f"[INDEX] Duplicates skipped: {duplicate_count}, new to add: {len(new_texts)}")
        
        if original_existing_hashes and new_hashes:
            new_hash_set = {h['hash'] for h in new_hashes}
            intersection = original_existing_hashes.intersection(new_hash_set)
            if intersection:
                new_texts_filtered = []
                new_hashes_filtered = []
                removed_count = 0
                
                for text, hash_info in zip(new_texts, new_hashes):
                    if hash_info['hash'] not in intersection:
                        new_texts_filtered.append(text)
                        new_hashes_filtered.append(hash_info)
                    else:
                        removed_count += 1
                
                new_texts = new_texts_filtered
                new_hashes = new_hashes_filtered
                duplicate_count += removed_count
        
        if not new_texts:
            print("[INDEX] All documents already in collection. Skip.")
            return
        
        print("[INDEX] Adding to vector store...")
        max_chars_per_doc = int(os.getenv("OLLAMA_EMBED_MAX_DOC_CHARS", "500"))

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
            print(f"[INDEX] {len(new_texts)} chunks → {len(expanded)} docs (trim to {max_chars_per_doc} chars each)")

        hashes_written = {doc.metadata.get("content_hash") for doc in expanded if doc.metadata.get("content_hash")}

        total_added = 0
        report_every = 100
        for doc in expanded:
            self.vector_store.add_documents([doc])
            total_added += 1
            if total_added % report_every == 0:
                print(f"[INDEX] Added: {total_added}/{len(expanded)}")
        print(f"[INDEX] Added {total_added} documents")

        print("[INDEX] Verifying...")
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
            print(f"[INDEX] Hashes in DB: {found}/{len(hashes_written)}")
            if missing:
                print(f"[INDEX] Missing: {missing}")
        except Exception as e:
            print(f"[INDEX] Verify error: {e}")

        print("[INDEX] Done\n")

    def answer(self, query: str) -> str:
        """Run RAG: retrieve context from ChromaDB, then generate answer with LLM."""
        print(f"[ANSWER] Query: {query[:80]}...")
        response = self.chain.invoke(query)
        print(f"[ANSWER] Response ({len(response)} chars)")
        for line in textwrap.wrap(response, 80):
            print(f"[ANSWER] ", end="")
            for char in line:
                print(char.upper(), end="", flush=True)
                time.sleep(0.01)
            print()
        print()
        return response

    def status(self):
        """Print vector DB stats, sample docs, duplicate check, and config."""
        print("=" * 80)
        print("VECTOR DB STATUS (ChromaDB)")
        print("=" * 80)

        if not hasattr(self, "vector_store") or self.vector_store is None:
            print("Error: ChromaDB not initialized")
            return

        print("\n[STATUS] 1. CHROMADB")
        print("-" * 80)
        try:
            print(f"  Path: {self.chroma_persist_directory}")
            print(f"  Collection: {self.collection_name}")
        except Exception as e:
            print(f"  Error: {e}")
            return

        points_count = 0
        print(f"\n[STATUS] 2. COLLECTION '{self.collection_name}'")
        print("-" * 80)
        try:
            points_count = self.vector_store._collection.count()
            print(f"  Vector size: {self.vector_size}")
            print(f"  Documents: {points_count}")
        except Exception as e:
            print(f"  Error: {e}")
            return

        print("\n[STATUS] 3. SAMPLE DOCUMENTS")
        print("-" * 80)
        try:
            result = self.vector_store._collection.get(
                limit=10,
                include=["metadatas", "documents"],
            )
            ids = result.get("ids") or []
            metadatas = result.get("metadatas") or []
            documents = result.get("documents") or []
            print(f"  Total in collection: {points_count}")

            if ids:
                sources = {}
                for i in range(len(ids)):
                    meta = metadatas[i] if i < len(metadatas) else {}
                    if not isinstance(meta, dict):
                        meta = {}
                    inner = meta.get("metadata", meta)
                    source = inner.get("source", meta.get("source", "unknown")) if isinstance(inner, dict) else meta.get("source", "unknown")
                    sources[source] = sources.get(source, 0) + 1
                print("  By source:")
                for source, cnt in list(sources.items())[:10]:
                    print(f"    - {source}: {cnt} chunks")
            else:
                print("  Collection empty")
        except Exception as e:
            print(f"  Error: {e}")

        print("\n[STATUS] 4. DUPLICATES CHECK")
        print("-" * 80)
        try:
            if points_count == 0:
                print("  (empty, skip)")
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
                print(f"  With hash: {len(hash_to_points)}, without: {len(points_without_hash)}, duplicate groups: {len(duplicates)}")
                if duplicates:
                    ids_to_delete = []
                    for content_hash, points_list in duplicates.items():
                        ids_to_delete.extend([p["id"] for p in points_list[1:]])
                    if ids_to_delete:
                        batch_size = 100
                        deleted_count = 0
                        for i in range(0, len(ids_to_delete), batch_size):
                            batch_ids = ids_to_delete[i : i + batch_size]
                            try:
                                self.vector_store.delete(ids=batch_ids)
                                deleted_count += len(batch_ids)
                            except Exception as e:
                                print(f"  Delete error: {e}")
                        print(f"  Deleted {deleted_count} duplicates")
                else:
                    print("  No duplicates")
        except Exception as e:
            print(f"  Error: {e}")
        
        print("\n[STATUS] 5. DOCS DIRECTORY (incl. subdirs)")
        print("-" * 80)
        try:
            if os.path.exists(self.index_path):
                md_files = []
                for root, _dirs, files in os.walk(self.index_path):
                    for file in files:
                        if file.endswith(".md"):
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, self.index_path)
                            md_files.append((rel_path, file_path))
                print(f"  Path: {self.index_path}")
                print(f"  .md files: {len(md_files)}")
                total_size = sum(os.path.getsize(fp) for _, fp in md_files)
                for rel_path, file_path in sorted(md_files)[:15]:
                    size = os.path.getsize(file_path)
                    print(f"    {rel_path}: {size:,} B")
                if len(md_files) > 15:
                    print(f"    ... and {len(md_files) - 15} more")
                print(f"  Total: {total_size:,} B ({total_size/1024:.1f} KB)")
            else:
                print(f"  Path missing: {self.index_path}")
        except Exception as e:
            print(f"  Error: {e}")
        
        print("\n[STATUS] 6. EMBEDDINGS (Ollama)")
        print("-" * 80)
        print(f"  Model: {self.embedding_model}, dim: {self.vector_size}")
        print(f"  URL: {self.ollama_base_url}")

        print("\n[STATUS] 7. LLM (Ollama)")
        print("-" * 80)
        print(f"  Model: {self.llm_model}, temperature: 0")
        
        print("\n[STATUS] 8. RETRIEVER")
        print("-" * 80)
        print(f"  k: {self.retriever.search_kwargs.get('k', 'N/A')}")
        
        print("\n[STATUS] 9. TEST SEARCH")
        print("-" * 80)
        try:
            if points_count > 0:
                results = self.retriever.get_relevant_documents("Python")
                print(f"  Query 'Python': {len(results)} results")
            else:
                print("  (empty)")
        except Exception as e:
            print(f"  Error: {e}")
        
        print("\n" + "=" * 80)
        print("STATUS DONE")
        print("=" * 80 + "\n")

    def clear(self):
        """Delete the ChromaDB collection and recreate it empty."""
        print("[CLEAR] Clearing ChromaDB collection...")
        try:
            self.vector_store._client.delete_collection(name=self.collection_name)
        except Exception as e:
            print(f"[CLEAR] Warning: {e}")
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.chroma_persist_directory,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        self.chain = (
            {"context": self.retriever | RunnableLambda(self._format_docs), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        print("[CLEAR] Done")

    def close(self):
        """No-op; ChromaDB persists to disk."""
        pass
    