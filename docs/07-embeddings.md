# 07. Эмбеддинги

Модуль `shkerebert/embeddings.py`.

## Что такое эмбеддинг в этом проекте

**Dense embedding** — фиксированный вектор (384 числа для MiniLM), представляющий
семантику текста. Похожие по смыслу тексты → близкие векторы (высокий cosine).

Используется библиотека **sentence-transformers** — обёртка над Transformer, которая
усредняет token embeddings в один sentence vector (mean pooling + нормализация).

## Модель по умолчанию

`sentence-transformers/all-MiniLM-L6-v2`:

- 6 слоёв, 384 измерения;
- обучена на парах предложений (semantic similarity);
- быстрая на CPU (~сотни предложений/сек в batch);
- англоязычная, но частично работает на других языках.

Многоязычный вариант (stretch): `paraphrase-multilingual-MiniLM-L12-v2` (384-d, 12 слоёв).

## Класс Embedder

```34:52:shkerebert/embeddings.py
class Embedder:
    def __init__(self, model_name: str, normalize: bool = True):
        self.model_name = model_name
        self.normalize = normalize
        self.model = _load_st_model(model_name)

    def encode(self, texts, batch_size=64, show_progress=False) -> np.ndarray:
        vecs = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            ...
        )
        return vecs.astype("float32")
```

### L2-нормализация

При `normalize=True` каждый вектор имеет длину 1. Тогда:

```
cosine(a, b) = dot(a, b)   // для unit vectors
```

Это позволяет использовать FAISS `IndexFlatIP` (inner product) как exact cosine search.

## Загрузка модели (offline-first)

```21:31:shkerebert/embeddings.py
@lru_cache(maxsize=2)
def _load_st_model(model_name: str):
    try:
        return SentenceTransformer(model_name, device="cpu", local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name, device="cpu")
```

1. Сначала пробует локальный HF-кэш без сети.
2. При отсутствии — качает онлайн.
3. `device="cpu"` — явно, даже если torch собран с CUDA.

`@lru_cache` — модель грузится один раз на процесс.

## Дисковый кэш эмбеддингов

На CPU эмбеддинг всего корпуса — самая дорогая операция при индексации. Корпус SQuAD
статичен → кэшируем.

### Сигнатура корпуса

```55:64:shkerebert/embeddings.py
def _corpus_signature(chunks, cfg):
    h = hashlib.sha1()
    h.update(cfg.retriever.embed_model.encode())
    h.update(f"{cfg.chunk.size}-{cfg.chunk.overlap}-{cfg.chunk.strategy}".encode())
    h.update(str(len(chunks)).encode())
    for ch in chunks:
        h.update(ch.id.encode())
    return h.hexdigest()[:16]
```

Ключ кэша меняется при смене: модели, параметров чанкинга, состава корпуса.

### embed_corpus

```67:94:shkerebert/embeddings.py
def embed_corpus(chunks, cfg, show_progress=True):
    embedder = Embedder(...)
    sig = _corpus_signature(chunks, cfg)
    vec_path = cache / f"{sig}.npy"
    meta_path = cache / f"{sig}.json"

    if vec_path.exists() and meta_path.exists():
        vecs = np.load(vec_path)
        if vecs.shape[0] == len(chunks):
            return vecs, embedder

    vecs = embedder.encode([c.text for c in chunks], ...)
    np.save(vec_path, vecs)
    meta_path.write_text(json.dumps({...}))
```

Файлы кэша:

- `{SHKEREBERT_DATA}/embeddings/{sig}.npy` — матрица `[N, dim] float32`;
- `{sig}.json` — метаданные (модель, n_chunks, dim, chunk_cfg).

Проверка `vecs.shape[0] == len(chunks)` — защита от частично записанного кэша.

## Encode вопроса vs корпуса

Один и тот же `Embedder.encode()` для:

- всех чанков при индексации (batch, show_progress);
- одного вопроса при retrieval (`retriever.retrieve`);
- батча вопросов в eval (`eval_retrieval.py`, `eval_e2e.py`).

Симметричный bi-encoder: question и passage кодируются одной моделью.

## Связь с requirements.txt

```6:9:requirements.txt
sentence-transformers>=2.7,<5
sentencepiece>=0.2          # токенизатор многоязычных моделей
```

Версия `<5` — в 5.x загрузчик через AutoProcessor ломает офлайн-загрузку
multilingual-модели; 4.x грузит через AutoTokenizer.
