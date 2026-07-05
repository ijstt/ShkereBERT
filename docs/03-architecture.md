# 03. Архитектура

## Схема end-to-end

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Documents   │───▶│   Chunking   │───▶│  Embeddings  │───▶│ FAISS Index  │
│ SQuAD/TXT/PDF│    │ sentence/    │    │ MiniLM L2    │    │ IndexFlatIP  │
└──────────────┘    │ fixed+overlap│    └──────────────┘    └──────┬───────┘
                    └──────────────┘                               │
                                                                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Answer+Src   │◀───│   Reader     │◀───│  Retriever   │◀───│ Query Embed  │
│ + Abstention │    │ ext/gen/hyb  │    │ top-k+BM25   │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
         ▲                  ▲
         │                  │
    pipeline.py         reader.py / generator.py
```

## Поток данных при `answer(question)`

1. **Retrieve** — `DenseRetriever.retrieve(question, k)`:
   - encode question → FAISS search → (опционально BM25 RRF, reranker) → `list[RetrievedChunk]`.
2. **Abstention L1** — если `retrieved[0].dense_score < min_score` → пустой ответ.
3. **Read** — в зависимости от `mode`:
   - `extractive`: `Reader.read()` на каждом чанке → лучший span;
   - `generative`: `Generator.generate()` с пронумерованными фрагментами;
   - `hybrid`: оба + верификация.
4. **Abstention L2** — reader-level (τ или маркер отказа).
5. **Return** — `Answer` с `sources: list[Source]`.

## Модули и ответственность

| Модуль | Классы / функции | Вход → выход |
|--------|------------------|--------------|
| `config.py` | `Config`, `ChunkConfig`, ... | Параметры пайплайна |
| `ingest.py` | `Document`, `load_squad_documents` | Файл/HF → `Document` |
| `chunking.py` | `Chunk`, `chunk_documents` | `Document` → `list[Chunk]` |
| `embeddings.py` | `Embedder`, `embed_corpus` | `list[Chunk]` → `ndarray[N,d]` |
| `index.py` | `VectorIndex` | embeddings + chunks → search |
| `retriever.py` | `DenseRetriever` | question → `RetrievedChunk[]` |
| `reader.py` | `Reader`, `SpanAnswer` | question + chunks → span |
| `generator.py` | `Generator`, `GenAnswer` | question + chunks → text |
| `pipeline.py` | `RAGPipeline`, `Answer` | question → Answer |
| `cli.py` | Typer commands | UX |

## Сборка пайплайна

```61:67:shkerebert/retriever.py
    @classmethod
    def build(cls, docs, cfg: Config, show_progress: bool = True) -> "DenseRetriever":
        """Собрать retriever из документов: чанкинг -> эмбеддинги -> индекс."""
        chunks = chunk_documents(docs, cfg.chunk, cfg.retriever.embed_model)
        embeddings, embedder = embed_corpus(chunks, cfg, show_progress=show_progress)
        index = VectorIndex(chunks, embeddings)
        return cls(index, embedder, cfg)
```

`RAGPipeline.build(docs, cfg)` вызывает `DenseRetriever.build` и оборачивает в пайплайн.
Reader и Generator загружаются **лениво** — Qwen не грузится в extractive-режиме:

```65:77:shkerebert/pipeline.py
    @property
    def reader(self) -> Reader:
        if self._reader is None:
            self._reader = Reader(self.cfg.reader)
        return self._reader

    @property
    def generator(self):
        if self._generator is None:
            from .generator import Generator
            self._generator = Generator(self.cfg.generator)
        return self._generator
```

## Модель данных

### Document (ingest)

```19:24:shkerebert/ingest.py
@dataclass(frozen=True)
class Document:
    id: str
    text: str
    title: str = ""
    meta: dict = field(default_factory=dict)
```

ID — SHA1-хэш текста с префиксом (`sq-`, `txt-`, `pdf-`).

### Chunk (chunking)

```29:37:shkerebert/chunking.py
@dataclass(frozen=True)
class Chunk:
    id: str          # "{doc_id}::{index}"
    doc_id: str
    text: str
    index: int
    n_tokens: int
    title: str = ""
    meta: dict = field(default_factory=dict)
```

### RetrievedChunk (retriever)

```31:35:shkerebert/retriever.py
@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float          # финальный скор (dense или rerank или RRF)
    dense_score: float    # исходный dense-скор (для абстенции)
```

**Важно:** `dense_score` всегда в шкале косинуса (inner product L2-нормированных векторов),
даже при BM25-фьюзе — на нём работает retrieval-level абстенция.

### Answer (pipeline)

```34:42:shkerebert/pipeline.py
@dataclass
class Answer:
    question: str
    answer: str                 # пустая строка => «нет ответа»
    is_answerable: bool
    confidence: float           # extractive: span-null; generative: NaN
    reason: str                 # почему так решили
    mode: str = "extractive"
    sources: list[Source] = field(default_factory=list)
```

## Оценочный контур (eval/)

Отдельно от продуктового пайплайна — скрипты в `eval/`:

- `build_corpus.py` — честный eval-корпус (уникальные контексты вопросов);
- `eval_retrieval.py` — Recall@k, MRR;
- `eval_e2e.py` — EM/F1 + калибровка τ;
- `ablations.py` — свип chunk_size / top_k;
- `eval_longdoc.py` — длинные документы;
- `eval_multiseed.py` — доверительные интервалы;
- `eval_retrieval_variants.py` — dense vs BM25 vs reranker;
- `eval_generative.py` — сравнение режимов ридера.

Результаты → `eval/results/*.csv` и `*.png`.

## Зависимости между слоями

```
Config
  ├── chunk.size, overlap, strategy  → chunking
  ├── retriever.embed_model, top_k   → embeddings, retriever
  ├── retriever.use_bm25, use_reranker → retriever (stretch)
  ├── reader.no_answer_threshold     → pipeline (extractive abstention)
  └── generator.model_path           → generator (lazy)
```

Смена `chunk.size` или `embed_model` инвалидирует кэш эмбеддингов (новая сигнатура в
`_corpus_signature`).
