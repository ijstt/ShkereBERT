# Векторный индекс (FAISS)

## Выбор: IndexFlatIP (Exact Search)

```python
import faiss
index = faiss.IndexFlatIP(dim)  # Inner Product
index.add(embeddings.astype("float32"))  # L2-нормированные!
```

### Почему не HNSW / IVF?

| Фактор | IndexFlatIP | HNSW / IVF |
|---|---|---|
| **Размер корпуса** | ≤ 50k чанков | 100k+ |
| **Время построения** | 0 мс (нет обучения) | Секунды–минуты |
| **Recall** | 1.0 (точный) | < 1.0 (приближённый) |
| **Память** | O(N·d) | O(N·d) + граф/квантизация |
| **Латентность поиска** | ~1–5 мс на 10k | ~0.5–1 мс |

**Наш корпус**: SQuAD validation ≈ 150 контекстов → ~1–3k чанков при chunk=256.
Точный перебор занимает **миллисекунды** и даёт идеальный recall. ANN не нужен.

### План масштабирования (100k+ чанков)

```python
# IVF (Inverted File) — кластеризация центроидов
nlist = 100  # число кластеров
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
index.train(embeddings)  # обучение центроидов
index.add(embeddings)
index.nprobe = 10  # сколько кластеров проверять при поиске

# Или HNSW (граф) — лучше recall/latency
index = faiss.IndexHNSWFlat(dim, 32)  # M=32 связей
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 128
index.add(embeddings)
```

В `shkerebert/index.py` класс `VectorIndex` инкапсулирует FAISS; замена индекса —
одна строка в `build()`.

## L2-нормировка = Cosine Similarity

```python
# В embeddings.py
vecs = model.encode(texts, normalize_embeddings=True)  # L2-норма
# Тогда:
#   cosine(u, v) = u·v / (|u||v|) = u·v  (так как |u|=|v|=1)
# FAISS IndexFlatIP считает inner product → это и есть cosine
```

**Почему это быстро**: не нужно считать sqrt(|u||v|) при поиске; нормализация
один раз при индексации.

## Сохранение и загрузка

```python
# Сохранение
faiss.write_index(index, "index.faiss")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

# Загрузка
index = faiss.read_index("index.faiss")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
```

В `VectorIndex.save()` / `load()` — атомарное сохранение индекса + чанков.
Ключ кэша эмбеддингов включает сигнатуру корпуса → при смене чанкинга/модели
индекс пересобирается автоматически.

## SearchHit

```python
@dataclass
class SearchHit:
    chunk: Chunk
    score: float  # cosine similarity ∈ [-1, 1], на практике [0, 1] для нормализованных
```

`retrieve()` возвращает `List[SearchHit]`, отсортированный по убыванию score.
