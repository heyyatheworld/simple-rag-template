import os
import time
import textwrap
import hashlib
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue, PointIdsList

class RAGPipeline:
    """
    RAGPipeline is a class that provides a pipeline for RAG (Retrieval-Augmented Generation) tasks.
    It is used to index a file and answer questions about the file.
    """
    def __init__(self):
        print()
        print("[INIT] Инициализация RAG Pipeline...")
        self.index_path = os.path.join(os.path.dirname(__file__), "docs")
        self.vectors = os.path.join(os.path.dirname(__file__), "vectors")
        print(f"[INIT] Путь к документам: {self.index_path}")
        print(f"[INIT] Путь к векторам: {self.vectors}")
        print()

        print("[INIT] Создание OpenAI Embeddings (text-embedding-3-small)...")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

        print("[INIT] Подключение к Qdrant (http://localhost:6333)...")
        self.client = QdrantClient(url="http://localhost:6333")

        print("[INIT] Проверка существующих коллекций...")
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        print(f"[INIT] Найдено коллекций: {collection_names}")
        
        collection_exists = any(c.name == "rag_collection" for c in collections)
        
        if collection_exists:
            print("[INIT] Коллекция 'rag_collection' найдена, проверка конфигурации...")
            try:
                collection_info = self.client.get_collection("rag_collection")
                vector_size = collection_info.config.params.vectors.size
                print(f"[INIT] Текущая размерность коллекции: {vector_size}")
                if vector_size != 1536:
                    print(f"[INIT] ВНИМАНИЕ: Размерность коллекции ({vector_size}) не соответствует требуемой (1536)")
                    print("[INIT] Коллекция будет использована, но возможны ошибки при работе с embeddings")
                else:
                    print("[INIT] Размерность коллекции соответствует требованиям")
                print("[INIT] Использование существующей коллекции 'rag_collection'")
            except Exception as e:
                print(f"[INIT] Ошибка при проверке коллекции: {e}")
                print("[INIT] Будет создана новая коллекция")
                collection_exists = False
        else:
            print("[INIT] Коллекция 'rag_collection' не найдена")
        if not collection_exists:
            print("[INIT] Создание коллекции 'rag_collection' с размерностью 1536...")
            self.client.create_collection(
                collection_name="rag_collection",
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE))

        print("[INIT] Инициализация QdrantVectorStore...")
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name="rag_collection",
            embedding=self.embeddings)

        print("[INIT] Создание retriever (k=5)...")
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        print("[INIT] Создание PromptTemplate...")
        self.prompt = PromptTemplate.from_template(
            "Answer the following question based on the context:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:")
        
        print("[INIT] Создание LLM (gpt-3.5-turbo)...")
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

        def format_docs(docs):
            print(f"[FORMAT_DOCS] Получено документов: {len(docs)}")
            for i, doc in enumerate(docs):
                content_preview = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
                print(f"[FORMAT_DOCS] Документ {i+1}: {len(doc.page_content)} символов, превью: {content_preview}")
            formatted = "\n\n".join(doc.page_content for doc in docs)
            print(f"[FORMAT_DOCS] Объединенный текст: {len(formatted)} символов")
            print()
            return formatted
        
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
            scroll_result = self.client.scroll(
                collection_name="rag_collection",
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            existing_points = scroll_result[0]
            
            for i, point in enumerate(existing_points):
                metadata = point.payload.get('metadata', {})
                if isinstance(metadata, dict) and 'content_hash' in metadata:
                    hash_value = metadata['content_hash']
                    existing_hashes.add(hash_value)
                    original_existing_hashes.add(hash_value)
                    existing_points_with_hash.append({
                        'id': point.id,
                        'hash': hash_value,
                        'source': metadata.get('source', point.payload.get('source', 'unknown'))
                    })
                else:
                    existing_points_without_hash.append({
                        'id': point.id,
                        'source': metadata.get('source', point.payload.get('source', 'unknown')) if isinstance(metadata, dict) else point.payload.get('source', 'unknown')
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
            scroll_result = self.client.scroll(
                collection_name="rag_collection",
                limit=len(new_texts) * 2,
                with_payload=True,
                with_vectors=False
            )
            all_points = scroll_result[0]
            
            found_hashes = set()
            missing_hashes = []
            
            for hash_val in hashes_before_write:
                found = False
                for point in all_points:
                    metadata = point.payload.get('metadata', {})
                    if isinstance(metadata, dict) and metadata.get('content_hash') == hash_val:
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
            
            if all_points:
                example_point = all_points[0]
                print(f"[INDEX] Пример структуры payload из базы:")
                print(f"[INDEX]   Ключи в payload: {list(example_point.payload.keys())}")
                metadata = example_point.payload.get('metadata', {})
                if isinstance(metadata, dict):
                    print(f"[INDEX]   Ключи в metadata: {list(metadata.keys())}")
                    if 'content_hash' in metadata:
                        print(f"[INDEX]   ✓ content_hash присутствует в metadata: {metadata['content_hash'][:16]}...")
                    else:
                        print(f"[INDEX]   ✗ content_hash ОТСУТСТВУЕТ в metadata!")
                else:
                    print(f"[INDEX]   ✗ metadata отсутствует или не является словарем!")
                    if 'content_hash' in example_point.payload:
                        print(f"[INDEX]   ✓ content_hash присутствует напрямую в payload: {example_point.payload['content_hash'][:16]}...")
                    else:
                        print(f"[INDEX]   ✗ content_hash ОТСУТСТВУЕТ в payload!")
                        print(f"[INDEX]   Доступные ключи: {list(example_point.payload.keys())}")
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
        print("ДИАГНОСТИКА ВЕКТОРНОЙ БАЗЫ ДАННЫХ")
        print("=" * 80)
        
        if not hasattr(self, 'client') or self.client is None:
            print("✗ Ошибка: Qdrant client не инициализирован")
            return
        
        print("\n[STATUS] 1. ПОДКЛЮЧЕНИЕ К QDRANT")
        print("-" * 80)
        try:
            collections = self.client.get_collections().collections
            print(f"✓ Подключение к Qdrant установлено (http://localhost:6333)")
            print(f"✓ Найдено коллекций: {len(collections)}")
            for col in collections:
                print(f"  - {col.name}")
        except Exception as e:
            print(f"✗ Ошибка подключения к Qdrant: {e}")
            return
        
        print("\n[STATUS] 2. КОЛЛЕКЦИЯ 'rag_collection'")
        print("-" * 80)
        try:
            collection_info = self.client.get_collection("rag_collection")
            config = collection_info.config
            params = config.params
            
            print(f"✓ Коллекция существует")
            print(f"  Размерность векторов: {params.vectors.size}")
            print(f"  Метрика расстояния: {params.vectors.distance}")
            print(f"  Количество шардов: {params.shard_number}")
            print(f"  Фактор репликации: {params.replication_factor}")
            
            collection_stats = self.client.get_collection("rag_collection")
            points_count = collection_stats.points_count
            indexed_vectors_count = collection_stats.indexed_vectors_count
            
            print(f"\n  Статистика:")
            print(f"  - Всего точек: {points_count}")
            print(f"  - Проиндексировано векторов: {indexed_vectors_count}")
            
            if hasattr(config, 'hnsw_config') and config.hnsw_config:
                hnsw = config.hnsw_config
                print(f"\n  HNSW конфигурация:")
                print(f"  - M: {hnsw.m}")
                print(f"  - ef_construct: {hnsw.ef_construct}")
                print(f"  - full_scan_threshold: {hnsw.full_scan_threshold}")
        except Exception as e:
            print(f"✗ Ошибка при получении информации о коллекции: {e}")
            return
        
        print("\n[STATUS] 3. ДОКУМЕНТЫ В КОЛЛЕКЦИИ")
        print("-" * 80)
        try:
            scroll_result = self.client.scroll(
                collection_name="rag_collection",
                limit=10,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            print(f"✓ Найдено документов в коллекции: {points_count}")
            
            if points:
                print(f"\n  Примеры документов (первые {min(10, len(points))}):")
                
                sources = {}
                for point in points:
                    metadata = point.payload.get('metadata', {})
                    if isinstance(metadata, dict):
                        source = metadata.get('source', point.payload.get('source', 'unknown'))
                    else:
                        source = point.payload.get('source', 'unknown')
                    if source not in sources:
                        sources[source] = []
                    sources[source].append(point)
                
                print(f"\n  Документы по источникам:")
                for source, source_points in sources.items():
                    print(f"    - {source}: {len(source_points)} чанков")
                
                print(f"\n  Примеры метаданных:")
                for i, point in enumerate(points[:3], 1):
                    payload = point.payload
                    metadata = payload.get('metadata', {})
                    if isinstance(metadata, dict):
                        source = metadata.get('source', payload.get('source', 'N/A'))
                        headers = {k: v for k, v in metadata.items() if k.startswith('Header')}
                    else:
                        # Fallback для старого формата
                        source = payload.get('source', 'N/A')
                        headers = {k: v for k, v in payload.items() if k.startswith('Header')}
                    
                    content_preview = payload.get('page_content', '')[:100] + "..." if len(payload.get('page_content', '')) > 100 else payload.get('page_content', '')
                    print(f"    Документ {i}:")
                    print(f"      ID: {point.id}")
                    print(f"      Источник: {source}")
                    if headers:
                        print(f"      Заголовки: {headers}")
                    print(f"      Размер контента: {len(payload.get('page_content', ''))} символов")
                    print(f"      Превью: {content_preview}")
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
                all_points = []
                offset = None
                batch_size = 1000
                
                while True:
                    scroll_result = self.client.scroll(
                        collection_name="rag_collection",
                        limit=batch_size,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    batch_points, next_offset = scroll_result
                    all_points.extend(batch_points)
                    
                    if next_offset is None:
                        break
                    offset = next_offset
                
                print(f"  Получено документов для проверки: {len(all_points)}")
                
                hash_to_points = {}
                points_without_hash = []
                
                for point in all_points:
                    metadata = point.payload.get('metadata', {})
                    content_hash = metadata.get('content_hash') if isinstance(metadata, dict) else None
                    if content_hash:
                        if content_hash not in hash_to_points:
                            hash_to_points[content_hash] = []
                        hash_to_points[content_hash].append(point)
                    else:
                        points_without_hash.append(point)
                
                duplicates = {hash_val: points_list for hash_val, points_list in hash_to_points.items() 
                             if len(points_list) > 1}
                
                print(f"  Документов с хешем: {len(hash_to_points)}")
                print(f"  Документов без хеша: {len(points_without_hash)}")
                print(f"  Найдено дубликатов (hash с несколькими документами): {len(duplicates)}")
                
                if duplicates:
                    total_duplicates_to_remove = 0
                    ids_to_delete = []
                    
                    for content_hash, duplicate_points in duplicates.items():
                        keep_point = duplicate_points[0]
                        points_to_remove = duplicate_points[1:]
                        
                        total_duplicates_to_remove += len(points_to_remove)
                        ids_to_delete.extend([point.id for point in points_to_remove])
                        
                        print(f"    Hash {content_hash[:16]}...: {len(duplicate_points)} документов, "
                              f"оставляем 1, удаляем {len(points_to_remove)}")
                    
                    if ids_to_delete:
                        print(f"\n  Удаление {len(ids_to_delete)} дубликатов...")
                        batch_delete_size = 100
                        deleted_count = 0
                        
                        for i in range(0, len(ids_to_delete), batch_delete_size):
                            batch_ids = ids_to_delete[i:i + batch_delete_size]
                            try:
                                self.client.delete(
                                    collection_name="rag_collection",
                                    points_selector=PointIdsList(points=batch_ids)
                                )
                                deleted_count += len(batch_ids)
                                print(f"    Удалено: {deleted_count}/{len(ids_to_delete)}")
                            except (TypeError, AttributeError) as e:
                                try:
                                    self.client.delete(
                                        collection_name="rag_collection",
                                        points_selector=batch_ids
                                    )
                                    deleted_count += len(batch_ids)
                                    print(f"    Удалено: {deleted_count}/{len(ids_to_delete)}")
                                except Exception as e2:
                                    print(f"    Ошибка при удалении батча: {e2}")
                                    for point_id in batch_ids:
                                        try:
                                            self.client.delete(
                                                collection_name="rag_collection",
                                                points_selector=[point_id]
                                            )
                                            deleted_count += 1
                                        except Exception as e3:
                                            print(f"      Ошибка при удалении точки {point_id}: {e3}")
                            except Exception as e:
                                print(f"    Ошибка при удалении батча: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        print(f"  ✓ Успешно удалено {deleted_count} дубликатов")
                        print(f"  ✓ Осталось уникальных документов: {len(all_points) - deleted_count}")
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
        
        print("\n[STATUS] 6. КОНФИГУРАЦИЯ EMBEDDINGS")
        print("-" * 80)
        try:
            print(f"✓ Модель: text-embedding-3-small")
            print(f"✓ Размерность: 1536")
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
                print(f"✓ API ключ установлен: {masked_key}")
            else:
                print(f"✗ API ключ не установлен")
        except Exception as e:
            print(f"✗ Ошибка: {e}")
        
        print("\n[STATUS] 7. КОНФИГУРАЦИЯ LLM")
        print("-" * 80)
        try:
            print(f"✓ Модель: gpt-3.5-turbo")
            print(f"✓ Temperature: 0")
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
                print(f"✓ API ключ установлен: {masked_key}")
            else:
                print(f"✗ API ключ не установлен")
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
        print("[CLEAR] Начало очистки коллекции...")
        self.client.delete_collection(collection_name="rag_collection")
        self.client.create_collection(
                collection_name="rag_collection",
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE))
        print("[CLEAR] Коллекция очищена успешно")
        print()
        
    def close(self):
        print("[CLOSE] Закрытие соединения с Qdrant...")
        self.client.close()
        print("[CLOSE] Соединение с Qdrant закрыто")
        print()
    