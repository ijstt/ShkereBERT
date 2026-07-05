# Документация ShkereBERT

Полное техническое описание проекта: архитектура, технологии, код и эксперименты.
Документация написана так, чтобы куратор мог проверить понимание каждого слоя — от
dataclass-конфига до калибровки порога абстенции на SQuAD v2.

## Оглавление

| № | Документ | О чём |
|---|----------|-------|
| 01 | [Обзор проекта](01-overview.md) | Цель, RAG, стек, режимы ридера, ограничения |
| 02 | [Установка и запуск](02-installation.md) | venv, модели, CLI, Streamlit, `run.sh` |
| 03 | [Архитектура](03-architecture.md) | Схема пайплайна, поток данных, модули |
| 04 | [Конфигурация](04-configuration.md) | Frozen dataclasses, `with_overrides`, кэш |
| 05 | [Загрузка данных](05-data-ingestion.md) | SQuAD v2, TXT, PDF, `Document` |
| 06 | [Чанкинг](06-chunking.md) | Sentence-aware, fixed, overlap, токены |
| 07 | [Эмбеддинги](07-embeddings.md) | Sentence-Transformers, L2-норм, дисковый кэш |
| 08 | [FAISS-индекс](08-faiss-index.md) | IndexFlatIP, cosine, save/load |
| 09 | [Retrieval](09-retrieval.md) | Dense, BM25, RRF, reranker, абстенция |
| 10 | [Экстрактивный ридер](10-extractive-reader.md) | BERT-span, null-score, τ |
| 11 | [Генеративный ридер](11-generative-reader.md) | Qwen GGUF, промпты, цитаты |
| 12 | [Пайплайн и абстенция](12-pipeline.md) | `RAGPipeline`, hybrid, `Answer` |
| 13 | [Интерфейсы](13-interfaces.md) | Typer CLI, Streamlit UI |
| 14 | [Оценка качества](14-evaluation.md) | Все eval-скрипты, метрики, протокол |
| 15 | [Тесты и зависимости](15-testing-deps.md) | pytest, requirements.txt |
| 16 | [Экспериментальные результаты](16-results.md) | Цифры, ablations, выводы |

## Быстрая навигация по коду

```
shkerebert/
  config.py      — единый конфиг (Chunk, Retriever, Reader, Generator)
  ingest.py      — Document, SQuAD/TXT/PDF
  chunking.py    — Chunk, sentence/fixed стратегии
  embeddings.py  — Embedder, embed_corpus + кэш
  index.py       — VectorIndex (FAISS)
  retriever.py   — DenseRetriever, BM25, reranker
  reader.py      — extractive BERT-span
  generator.py   — Qwen через llama-cpp
  pipeline.py    — RAGPipeline, три режима
  cli.py         — Typer CLI

app/streamlit_app.py  — веб-UI
eval/                 — скрипты оценки + results/
tests/                — юнит-тесты
scripts/download_models.py — загрузка HF-моделей
```

## Связанные файлы в корне

- `README.md` — краткое описание и ключевые метрики
- `CHANGELOG.md` — хронология решений (для защиты)
- `models/README.md` — где лежит GGUF Qwen
- `run.sh` — обёртка для запуска
