# 08. FAISS-индекс

Модуль `shkerebert/index.py`.

## FAISS (Facebook AI Similarity Search)

Библиотека для быстрого поиска ближайших соседей в многомерном пространстве.
В проекте — **faiss-cpu** (без GPU).

## IndexFlatIP — точный inner product

```26:35:shkerebert/index.py
class VectorIndex:
    def __init__(self, chunks, embeddings):
        import faiss
        assert embeddings.shape[0] == len(chunks)
        self.chunks = chunks
        self.dim = int(embeddings.shape[1])
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(np.ascontiguousarray(embeddings.astype("float32")))
```

| Свойство | Значение |
|----------|----------|
| Тип | `IndexFlatIP` — **точный** (не ANN) |
| Метрика | Inner product |
| При L2-norm | = cosine similarity |
| Сложность | O(N) на запрос — перебор всех векторов |

**Почему Flat, а не IVF/HNSW:**

- Корпус SQuAD — десятки тысяч чанков, перебор занимает миллисекунды.
- Точность важнее скорости для учебной оценки.
- Нет параметров кластеризации / efSearch для подбора.

## Search

```37:50:shkerebert/index.py
    def search(self, query_vecs, k=5):
        if query_vecs.ndim == 1:
            query_vecs = query_vecs[None, :]
        q = np.ascontiguousarray(query_vecs.astype("float32"))
        scores, idx = self.index.search(q, k)
        ...
        hits = [SearchHit(chunk=self.chunks[i], score=float(s))
                for s, i in zip(row_scores, row_idx) if i != -1]
```

- Поддерживает batch-запросы (несколько вопросов за раз).
- `idx == -1` — padding при нехватке результатов (не должно случаться при k ≤ N).
- Возвращает `SearchHit(chunk, score)`.

## SearchHit

```20:23:shkerebert/index.py
@dataclass
class SearchHit:
    chunk: Chunk
    score: float
```

## Сохранение и загрузка

```53:72:shkerebert/index.py
    def save(self, directory):
        self._faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, directory):
        chunks = pickle.load(...)
        obj.index = faiss.read_index(...)
```

В текущем пайплайне индекс **пересобирается в памяти** при каждом `DenseRetriever.build`;
save/load — API для расширений (персистентный индекс между сессиями).

## reconstruct для BM25-фьюза

При RRF-слиянии кандидат может прийти только из BM25, без dense-скора. Retriever
реконструирует вектор из FAISS:

```python
dense[i] = float(np.dot(qvec, self.index.index.reconstruct(i)))
```

`reconstruct(i)` возвращает i-й вектор из индекса — dot с query = cosine (при norm).

## Размерность

Определяется эмбеддером: 384 для MiniLM / multilingual-MiniLM.
`self.dim = embeddings.shape[1]` — проверка согласованности.

## Зависимость

```11:11:requirements.txt
faiss-cpu>=1.8
```

Пакет `faiss-gpu` не используется — весь проект CPU-only.
