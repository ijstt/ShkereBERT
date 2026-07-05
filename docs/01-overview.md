# 01. Обзор проекта

## Что это

**ShkereBERT** — учебный RAG-чат-бот, который отвечает на вопросы по текстовому
документу (или корпусу документов). Система:

1. режет документы на фрагменты (chunks);
2. ищет top-k релевантных фрагментов по вопросу (retrieval);
3. формирует ответ одним из трёх ридеров;
4. возвращает **цитату источника** и умеет честно сказать **«ответа нет»**.

Валидация — на [SQuAD v2](https://huggingface.co/datasets/rajpurkar/squad_v2) по
официальным метрикам **Exact Match / F1** (включая подметрику NoAns для unanswerable
вопросов).

## RAG (Retrieval-Augmented Generation)

Классический QA без retrieval требует, чтобы модель «знала» весь документ в контексте.
Для длинных документов это невозможно: окно LLM/BERT ограничено (384–4096 токенов).

**RAG** разбивает задачу:

| Этап | Что делает | Модуль |
|------|------------|--------|
| Indexing | Документ → чанки → эмбеддинги → FAISS | `chunking`, `embeddings`, `index` |
| Retrieval | Вопрос → эмбеддинг → top-k чанков | `retriever` |
| Reading | Вопрос + чанки → ответ | `reader` / `generator` |

Пайплайн собран в `RAGPipeline`:

```51:94:shkerebert/pipeline.py
class RAGPipeline:
    def __init__(self, retriever: DenseRetriever, cfg: Config,
                 reader: Reader | None = None, generator=None):
        ...
    @classmethod
    def build(cls, docs, cfg: Config | None = None, show_progress: bool = True):
        cfg = cfg or default_config()
        retriever = DenseRetriever.build(docs, cfg, show_progress=show_progress)
        return cls(retriever, cfg)

    def answer(self, question: str, k: int | None = None, mode: str | None = None) -> Answer:
        ...
        retrieved: list[RetrievedChunk] = self.retriever.retrieve(question, k=k)
        # --- Уровень 1: retrieval-level абстенция ---
        if not retrieved or retrieved[0].dense_score < self.cfg.retriever.min_score:
            return Answer(question, "", False, ...)
        if mode == "generative":
            return self._answer_generative(question, retrieved)
        if mode == "hybrid":
            return self._answer_hybrid(question, retrieved)
        return self._answer_extractive(question, retrieved)
```

## Три режима ридера

| Режим | Модель | Скорость (CPU) | Назначение |
|-------|--------|----------------|------------|
| `extractive` | `deepset/tinyroberta-squad2` | ~230 мс/вопрос | Baseline, EM/F1-якорь, калиброванный τ |
| `generative` | Qwen 2.5 3B GGUF (llama-cpp) | ~10 с/вопрос | Естественный ответ + цитаты `[n]`, RU/EN |
| `hybrid` | Qwen + BERT-верификатор | ~10 с/вопрос | Анти-галлюцинация, fallback на extractive |

Переключение: `cfg.reader_mode` или аргумент `mode=` в `pipe.answer(...)`.

## Двухуровневая абстенция «нет ответа»

SQuAD v2 содержит вопросы, на которые **в контексте нет ответа** (`is_impossible`).
Система отказывается отвечать на двух уровнях:

1. **Retrieval-level**: лучший найденный фрагмент слишком непохож на вопрос
   (`dense_score < min_score`, по умолчанию 0.15).
2. **Reader-level**:
   - extractive: `gap = null_score - best_span_score > τ` (τ калибруется на dev);
   - generative: модель возвращает маркер `НЕТ ОТВЕТА` / `NO ANSWER`.

## Стек технологий

| Компонент | Библиотека / модель | Файл |
|-----------|---------------------|------|
| Эмбеддинги | `sentence-transformers/all-MiniLM-L6-v2` (384-d) | `embeddings.py` |
| Векторный поиск | `faiss-cpu`, IndexFlatIP | `index.py` |
| Экстрактивный QA | `transformers`, tinyroberta-squad2 | `reader.py` |
| Генерация | `llama-cpp-python`, Qwen 2.5 3B Q4_K_M | `generator.py` |
| BM25-гибрид | `rank-bm25`, RRF | `retriever.py` |
| Reranker | `CrossEncoder/ms-marco-MiniLM-L-6-v2` | `retriever.py` |
| Данные | `datasets` (SQuAD), `pypdf` (PDF) | `ingest.py` |
| Метрики | `evaluate` (squad_v2) | `eval/eval_e2e.py` |
| CLI | `typer` | `cli.py` |
| Web UI | `streamlit` | `app/streamlit_app.py` |

Все модели работают на **CPU** — on-premise сценарий без GPU.

## Ключевые метрики (baseline, held-out test)

| Метрика | Значение | Смысл |
|---------|----------|-------|
| F1 (e2e) | **75.8** | Полный пайплайн retrieval + reader |
| Oracle F1 | **81.0** | Reader на золотом контексте (потолок) |
| Retrieval loss | **5.3 F1** | Цена этапа поиска |
| Recall@5 | **0.942** | Золотой документ в top-5 |
| NoAns F1 | **84.6** | Качество отказа |
| Latency | **229 мс/q** | extractive, k=5 |

Подробнее — [16-results.md](16-results.md).

## Ограничения (честно)

- **Extractive reader** возвращает один непрерывный span — не собирает составной ответ
  из нескольких мест документа.
- **Домен**: baseline заточен под энциклопедический стиль SQuAD (Wikipedia EN).
- **τ** откалиброван на dev SQuAD v2; для нового корпуса нужна перекалибровка.
- **PDF**: pypdf не делает OCR — сканы без текстового слоя не читаются.
- **Русский**: stretch через `multilingual_config()` + generative Qwen; extractive reader
  остаётся англоязычным.
- **Retrieval видит только top-k чанков** — если ответ размазан по далёким фрагментам,
  контекст может быть неполным.

## Структура репозитория

```
shkerebert/   — Python-пакет (ядро)
app/          — Streamlit UI
eval/         — оценка + eval/results/
tests/        — pytest
demo/         — демо-документы (EN, RU)
scripts/      — download_models.py
models/       — GGUF Qwen (не в git)
docs/         — эта документация
```
