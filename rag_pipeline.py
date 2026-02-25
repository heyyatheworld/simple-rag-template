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
    """
    RAGPipeline is a class that provides a pipeline for RAG (Retrieval-Augmented Generation) tasks.
    It is used to index a file and answer questions about the file.
    """
    def __init__(self):
        print()
        print("[INIT] Инициализация RAG Pipeline...")

        # Конфигурация из .env
        self.collection_name = os.getenv("CHROMA_COLLECTION", "rag_collection")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self.llm_model = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
        self.vector_size = int(os.getenv("VECTOR_SIZE", "1024"))  # nomic-embed-text = 1024
        docs_path = os.getenv("DOCS_PATH", "docs")
        vectors_path = os.getenv("VECTORS_PATH", "vectors")
        chroma_persist = os.getenv("CHROMA_PERSIST_DIR", "").strip() or os.path.join(
            os.path.dirname(__file__), vectors_path, "chroma_db"
        )

        self.index_path = os.path.join(os.path.dirname(__file__), docs_path)
        self.vectors = os.path.join(os.path.dirname(__file__), vectors_path)
        self.chroma_persist_directory = chroma_persist
        print(f"[INIT] Путь к документам: {self.index_path}")
        print(f"[INIT] Путь к ChromaDB: {self.chroma_persist_directory}")
        print()

        print(f"[INIT] Создание Ollama Embeddings ({self.embedding_model})...")
        self.embeddings = OllamaEmbeddings(
            model=self.embedding_model,
            base_url=self.ollama_base_url,
        )

        print(f"[INIT] Инициализация ChromaDB (коллекция '{self.collection_name}')...")
        os.makedirs(self.chroma_persist_directory, exist_ok=True)
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.chroma_persist_directory,
            embedding_function=self.embeddings,
        )

        print("[INIT] Создание retriever (k=5)...")
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        print("[INIT] Создание PromptTemplate...")
        self.prompt = PromptTemplate.from_template(
            "Answer the following question based on the context:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        )

        print(f"[INIT] Создание LLM Ollama ({self.llm_model})...")
        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=0,
            base_url=self.ollama_base_url,
        )

        def format_docs(docs):
            print(f"[FORMAT_DOCS] Получено документов: {len(docs)}")
            for i, doc in enumerate(docs):
                content_preview = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
                print(f"[FORMAT_DOCS] Документ {i+1}: {len(doc.page_content)} символов, превью: {content_preview}")
            formatted = "\n\n".join(doc.page_content for doc in docs)
            print(f"[FORMAT_DOCS] Объединенный текст: {len(formatted)} символов")
            print()
            return formatted

        self._format_docs = format_docs
        print("[INIT] Создание RAG chain...")
        self.chain = (
            {"context": self.retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        print("[INIT] Инициализация завершена успешно\n")

    def index(self):
        print("[INDEX] Начало индексации документов...")
        print(f"[INDEX] Чтение файлов из: {self.index_path}")
        
        documents = []
        files = os.listdir(self.index_path)
        print(f"[INDEX] Найдено файлов: {len(files)}")
        
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(self.index_path, file)
                print(f"[INDEX] Чтение файла: {file}")
                with open(file_path, "r") as f:
                    content = f.read()
                    doc = Document(page_content=content, metadata={"source": file})
                    documents.append(doc)
                    print(f"[INDEX] Файл '{file}' прочитан: {len(content)} символов")
        print(f"[INDEX] Всего документов загружено: {len(documents)}")
        print()

        print("[INDEX] Разделение документов по markdown заголовкам...")

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),])
        
        split_docs = []
        for doc in documents:
            print(f"[INDEX] Разбиение документа '{doc.metadata.get('source', 'unknown')}' по заголовкам...")
            header_splits = markdown_splitter.split_text(doc.page_content)
            for split in header_splits:
                split.metadata.update(doc.metadata)
                headers = {k: v for k, v in split.metadata.items() if k.startswith("Header")}
                if headers:
                    print(f"[INDEX]   Раздел: {headers}, размер: {len(split.page_content)} символов")
            split_docs.extend(header_splits)
            print(f"[INDEX] Документ разбит на {len(header_splits)} разделов")
        
        print(f"[INDEX] После разбиения по заголовкам: {len(split_docs)} разделов")
        print()
        
        print("[INDEX] Создание RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=200)...")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n## ", "\n\n", "\n", ". ", " ", ""])
        print()
        print("[INDEX] Финальное разбиение больших разделов на чанки...")
        texts = text_splitter.split_documents(split_docs)
        print(f"[INDEX] Создано чанков: {len(texts)}")
        for i, text in enumerate(texts[:5]):
            preview = text.page_content[:100] + "..." if len(text.page_content) > 100 else text.page_content
            headers = {k: v for k, v in text.metadata.items() if k.startswith("Header")}
            header_info = f", заголовок: {headers}" if headers else ""
            #print(f"[INDEX] Чанк {i+1}: {len(text.page_content)} символов{header_info}")
            #print(f"[INDEX]   Превью: {preview}")
        if len(texts) > 5:
            print(f"[INDEX] ... и еще {len(texts) - 5} чанков")
        print()

        print("[INDEX] Проверка на дубликаты...")

        existing_hashes = set()
        original_existing_hashes = set()
        existing_points_with_hash = []
        existing_points_without_hash = []

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
                # LangChain Chroma может хранить metadata вложенно
                inner = meta.get("metadata", meta)
                if isinstance(inner, dict) and "content_hash" in inner:
                    hash_value = inner["content_hash"]
                    existing_hashes.add(hash_value)
                    original_existing_hashes.add(hash_value)
                    existing_points_with_hash.append({
                        "id": doc_id,
                        "hash": hash_value,
                        "source": inner.get("source", meta.get("source", "unknown")),
                    })
                elif isinstance(meta, dict) and "content_hash" in meta:
                    hash_value = meta["content_hash"]
                    existing_hashes.add(hash_value)
                    original_existing_hashes.add(hash_value)
                    existing_points_with_hash.append({
                        "id": doc_id,
                        "hash": hash_value,
                        "source": meta.get("source", "unknown"),
                    })
                else:
                    existing_points_without_hash.append({
                        "id": doc_id,
                        "source": meta.get("source", "unknown") if isinstance(meta, dict) else "unknown",
                    })

            print(f"[INDEX] Найдено существующих документов: {len(existing_hashes)}")
        except Exception as e:
            print(f"[INDEX] Ошибка при получении существующих документов: {e}")
            existing_hashes = set()
            original_existing_hashes = set()
        
        new_texts = []
        duplicate_count = 0
        new_hashes = []
        duplicate_hashes = []
        
        for idx, text in enumerate(texts, 1):
            source = text.metadata.get('source', 'unknown')
            
            content_to_hash = f"{text.page_content}{source}"
            
            content_hash = hashlib.md5(content_to_hash.encode('utf-8')).hexdigest()
            
            text.metadata['content_hash'] = content_hash
            
            if content_hash in existing_hashes:
                duplicate_count += 1
                duplicate_hashes.append({
                    'hash': content_hash,
                    'source': source,
                    'index': idx
                })
            else:
                new_texts.append(text)
                new_hashes.append({
                    'hash': content_hash,
                    'source': source,
                    'index': idx
                })
                existing_hashes.add(content_hash)
        
        print(f"[INDEX] Найдено дубликатов: {duplicate_count}")
        print(f"[INDEX] Новых документов для добавления: {len(new_texts)}")
        
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
            print("[INDEX] Все документы уже существуют в коллекции. Индексация не требуется.")
            print("[INDEX] Индексация завершена успешно\n")
            return
        
        print("[INDEX] Добавление новых документов в векторное хранилище...")
        
        hashes_before_write = []
        for text in new_texts:
            if 'content_hash' in text.metadata:
                hashes_before_write.append(text.metadata['content_hash'])
            else:
                print(f"[INDEX] ⚠ ВНИМАНИЕ: Документ без хеша перед записью!")
                print(f"[INDEX]   Source: {text.metadata.get('source', 'unknown')}")
                print(f"[INDEX]   Content preview: {text.page_content[:50]}...")
        
        print(f"[INDEX] Хешей в метаданных перед записью: {len(hashes_before_write)}/{len(new_texts)}")
        
        self.vector_store.add_documents(new_texts)
        print(f"[INDEX] {len(new_texts)} документов добавлено в векторное хранилище")
        
        print("[INDEX] Проверка сохранения хешей в базе...")
        try:
            result = self.vector_store._collection.get(
                include=["metadatas"],
                limit=len(new_texts) * 2 + 1000,
            )
            metadatas = result.get("metadatas") or []

            found_hashes = set()
            missing_hashes = []

            for hash_val in hashes_before_write:
                found = False
                for meta in metadatas:
                    if not isinstance(meta, dict):
                        continue
                    inner = meta.get("metadata", meta)
                    if isinstance(inner, dict) and inner.get("content_hash") == hash_val:
                        found_hashes.add(hash_val)
                        found = True
                        break
                    if meta.get("content_hash") == hash_val:
                        found_hashes.add(hash_val)
                        found = True
                        break
                if not found:
                    missing_hashes.append(hash_val)

            print(f"[INDEX] Хешей найдено в базе: {len(found_hashes)}/{len(hashes_before_write)}")
            if missing_hashes:
                print(f"[INDEX] ⚠ Хешей НЕ найдено в базе: {len(missing_hashes)}")
                print(f"[INDEX]   Примеры отсутствующих хешей:")
                for hash_val in missing_hashes[:5]:
                    print(f"[INDEX]     - {hash_val}")

            if metadatas:
                example = metadatas[0] if isinstance(metadatas[0], dict) else {}
                inner = example.get("metadata", example)
                print(f"[INDEX] Пример структуры метаданных в ChromaDB:")
                print(f"[INDEX]   Ключи: {list(inner.keys()) if isinstance(inner, dict) else list(example.keys())}")
                if isinstance(inner, dict) and "content_hash" in inner:
                    print(f"[INDEX]   ✓ content_hash присутствует: {inner['content_hash'][:16]}...")
                elif isinstance(example, dict) and "content_hash" in example:
                    print(f"[INDEX]   ✓ content_hash присутствует: {example['content_hash'][:16]}...")
                else:
                    print(f"[INDEX]   ✗ content_hash не найден в примере")
        except Exception as e:
            print(f"[INDEX] ⚠ Ошибка при проверке сохранения хешей: {e}")
            import traceback
            traceback.print_exc()
        
        print("[INDEX] Индексация завершена успешно\n")
        print()

    def answer(self, query: str) -> str:
        print(f"[ANSWER] Получен запрос: {query}")
        print(f"[ANSWER] Длина запроса: {len(query)} символов")
        
        print("[ANSWER] Запуск RAG chain...")
        print("[ANSWER] Поиск релевантных документов через retriever...")
        
        response = self.chain.invoke(query)

        print(f"[ANSWER] Получен ответ от LLM: {len(response)} символов")
        print()
        for line in textwrap.wrap(response, 80):
            print(f"[ANSWER] ", end="")
            for char in line:
                print(char.upper(), end="", flush=True)
                time.sleep(0.01)
            print()
        print()
        print("[ANSWER] Обработка запроса завершена\n")
        return response
    
    def status(self):
        print("=" * 80)
        print("ДИАГНОСТИКА ВЕКТОРНОЙ БАЗЫ ДАННЫХ (ChromaDB)")
        print("=" * 80)

        if not hasattr(self, "vector_store") or self.vector_store is None:
            print("✗ Ошибка: ChromaDB vector store не инициализирован")
            return

        print("\n[STATUS] 1. ПОДКЛЮЧЕНИЕ К CHROMADB")
        print("-" * 80)
        try:
            print(f"✓ Путь к данным: {self.chroma_persist_directory}")
            print(f"✓ Коллекция: {self.collection_name}")
        except Exception as e:
            print(f"✗ Ошибка: {e}")
            return

        points_count = 0
        print(f"\n[STATUS] 2. КОЛЛЕКЦИЯ '{self.collection_name}'")
        print("-" * 80)
        try:
            points_count = self.vector_store._collection.count()
            print(f"✓ Коллекция существует")
            print(f"  Размерность векторов: {self.vector_size} (из конфигурации)")
            print(f"\n  Статистика:")
            print(f"  - Всего документов: {points_count}")
        except Exception as e:
            print(f"✗ Ошибка при получении информации о коллекции: {e}")
            return

        print("\n[STATUS] 3. ДОКУМЕНТЫ В КОЛЛЕКЦИИ")
        print("-" * 80)
        try:
            result = self.vector_store._collection.get(
                limit=10,
                include=["metadatas", "documents"],
            )
            ids = result.get("ids") or []
            metadatas = result.get("metadatas") or []
            documents = result.get("documents") or []
            print(f"✓ Найдено документов в коллекции: {points_count}")

            if ids:
                print(f"\n  Примеры документов (первые {min(10, len(ids))}):")
                sources = {}
                for i, doc_id in enumerate(ids):
                    meta = metadatas[i] if i < len(metadatas) else {}
                    if not isinstance(meta, dict):
                        meta = {}
                    inner = meta.get("metadata", meta)
                    source = inner.get("source", meta.get("source", "unknown")) if isinstance(inner, dict) else meta.get("source", "unknown")
                    if source not in sources:
                        sources[source] = 0
                    sources[source] += 1
                print(f"\n  Документы по источникам:")
                for source, cnt in sources.items():
                    print(f"    - {source}: {cnt} чанков")
                print(f"\n  Примеры метаданных:")
                for i in range(min(3, len(ids))):
                    meta = metadatas[i] if i < len(metadatas) else {}
                    inner = meta.get("metadata", meta) if isinstance(meta, dict) else {}
                    doc_text = documents[i] if i < len(documents) else ""
                    source = inner.get("source", "N/A") if isinstance(inner, dict) else "N/A"
                    headers = {k: v for k, v in (inner or {}).items() if k.startswith("Header")} if isinstance(inner, dict) else {}
                    preview = (doc_text[:100] + "...") if len(doc_text) > 100 else doc_text
                    print(f"    Документ {i + 1}:")
                    print(f"      ID: {ids[i]}")
                    print(f"      Источник: {source}")
                    if headers:
                        print(f"      Заголовки: {headers}")
                    print(f"      Размер контента: {len(doc_text)} символов")
                    print(f"      Превью: {preview}")
            else:
                print("  ⚠ Коллекция пуста - документы не проиндексированы")
        except Exception as e:
            print(f"✗ Ошибка при получении документов: {e}")

        print("\n[STATUS] 4. ПРОВЕРКА НА ДУБЛИКАТЫ")
        print("-" * 80)
        try:
            if points_count == 0:
                print("  ⚠ Коллекция пуста - проверка на дубликаты пропущена")
            else:
                print("  Получение всех документов из коллекции...")
                result = self.vector_store._collection.get(
                    include=["metadatas"],
                    limit=100000,
                )
                all_ids = result.get("ids") or []
                all_metadatas = result.get("metadatas") or []
                print(f"  Получено документов для проверки: {len(all_ids)}")

                hash_to_points = {}
                points_without_hash = []

                for i, doc_id in enumerate(all_ids):
                    meta = all_metadatas[i] if i < len(all_metadatas) else {}
                    if not isinstance(meta, dict):
                        meta = {}
                    inner = meta.get("metadata", meta)
                    content_hash = (inner.get("content_hash") if isinstance(inner, dict) else None) or meta.get("content_hash")
                    if content_hash:
                        if content_hash not in hash_to_points:
                            hash_to_points[content_hash] = []
                        hash_to_points[content_hash].append({"id": doc_id, "meta": meta})
                    else:
                        points_without_hash.append(doc_id)

                duplicates = {h: lst for h, lst in hash_to_points.items() if len(lst) > 1}
                print(f"  Документов с хешем: {len(hash_to_points)}")
                print(f"  Документов без хеша: {len(points_without_hash)}")
                print(f"  Найдено дубликатов (hash с несколькими документами): {len(duplicates)}")

                if duplicates:
                    ids_to_delete = []
                    for content_hash, points_list in duplicates.items():
                        points_to_remove = points_list[1:]
                        ids_to_delete.extend([p["id"] for p in points_to_remove])
                        print(f"    Hash {content_hash[:16]}...: {len(points_list)} документов, "
                              f"оставляем 1, удаляем {len(points_to_remove)}")
                    if ids_to_delete:
                        print(f"\n  Удаление {len(ids_to_delete)} дубликатов...")
                        batch_size = 100
                        deleted_count = 0
                        for i in range(0, len(ids_to_delete), batch_size):
                            batch_ids = ids_to_delete[i : i + batch_size]
                            try:
                                self.vector_store.delete(ids=batch_ids)
                                deleted_count += len(batch_ids)
                                print(f"    Удалено: {deleted_count}/{len(ids_to_delete)}")
                            except Exception as e:
                                print(f"    Ошибка при удалении батча: {e}")
                        print(f"  ✓ Успешно удалено {deleted_count} дубликатов")
                        print(f"  ✓ Осталось уникальных документов: {len(all_ids) - deleted_count}")
                    else:
                        print("  ⚠ Нет дубликатов для удаления")
                else:
                    print("  ✓ Дубликаты не найдены - все документы уникальны")
        except Exception as e:
            print(f"✗ Ошибка при проверке на дубликаты: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n[STATUS] 5. ФАЙЛЫ В ДИРЕКТОРИИ ДОКУМЕНТОВ")
        print("-" * 80)
        try:
            if os.path.exists(self.index_path):
                files = os.listdir(self.index_path)
                print(f"✓ Директория существует: {self.index_path}")
                print(f"✓ Найдено файлов: {len(files)}")
                
                total_size = 0
                for file in files:
                    file_path = os.path.join(self.index_path, file)
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path)
                        total_size += size
                        print(f"  - {file}: {size:,} байт ({size/1024:.2f} KB)")
                
                print(f"\n  Общий размер: {total_size:,} байт ({total_size/1024:.2f} KB)")
            else:
                print(f"✗ Директория не существует: {self.index_path}")
        except Exception as e:
            print(f"✗ Ошибка при проверке директории: {e}")
        
        print("\n[STATUS] 6. КОНФИГУРАЦИЯ EMBEDDINGS (Ollama)")
        print("-" * 80)
        try:
            print(f"✓ Модель: {self.embedding_model}")
            print(f"✓ Размерность: {self.vector_size}")
            print(f"✓ Ollama URL: {self.ollama_base_url}")
        except Exception as e:
            print(f"✗ Ошибка: {e}")

        print("\n[STATUS] 7. КОНФИГУРАЦИЯ LLM (Ollama)")
        print("-" * 80)
        try:
            print(f"✓ Модель: {self.llm_model}")
            print(f"✓ Temperature: 0")
            print(f"✓ Ollama URL: {self.ollama_base_url}")
        except Exception as e:
            print(f"✗ Ошибка: {e}")
        
        print("\n[STATUS] 8. КОНФИГУРАЦИЯ RETRIEVER")
        print("-" * 80)
        try:
            search_kwargs = self.retriever.search_kwargs
            print(f"✓ Количество возвращаемых документов (k): {search_kwargs.get('k', 'N/A')}")
            print(f"✓ Тип поиска: векторный поиск по косинусному расстоянию")
        except Exception as e:
            print(f"✗ Ошибка: {e}")
        
        print("\n[STATUS] 9. ТЕСТОВЫЙ ПОИСК")
        print("-" * 80)
        try:
            if points_count > 0:
                test_query = "Python"
                print(f"  Тестовый запрос: '{test_query}'")
                results = self.retriever.get_relevant_documents(test_query)
                print(f"✓ Поиск выполнен успешно")
                print(f"  Найдено релевантных документов: {len(results)}")
                if results:
                    print(f"  Первый результат:")
                    first_result = results[0]
                    preview = first_result.page_content[:150] + "..." if len(first_result.page_content) > 150 else first_result.page_content
                    print(f"    Размер: {len(first_result.page_content)} символов")
                    print(f"    Превью: {preview}")
            else:
                print("  ⚠ Коллекция пуста - тестовый поиск пропущен")
        except Exception as e:
            print(f"✗ Ошибка при тестовом поиске: {e}")
        
        print("\n" + "=" * 80)
        print("ДИАГНОСТИКА ЗАВЕРШЕНА")
        print("=" * 80 + "\n")

    def clear(self):
        print("[CLEAR] Начало очистки коллекции ChromaDB...")
        try:
            self.vector_store._client.delete_collection(name=self.collection_name)
        except Exception as e:
            print(f"[CLEAR] Предупреждение при удалении коллекции: {e}")
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
        print("[CLEAR] Коллекция очищена успешно")
        print()

    def close(self):
        print("[CLOSE] ChromaDB хранит данные на диске, явное закрытие не требуется.")
        print()
    