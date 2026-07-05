# 15. Тесты и зависимости

## pytest

```bash
.venv/bin/python -m pytest -q
# или
./run.sh test
```

Каталог `tests/` — юнит- и интеграционные тесты без моков тяжёлых моделей там, где
используется реальный MiniLM (кэшируется после первого прогона).

---

## tests/test_chunking.py

| Тест | Что проверяет |
|------|---------------|
| `test_split_sentences_basic` | Regex-разбиение `.!?` |
| `test_split_sentences_handles_newlines` | Абзацы |
| `test_chunk_respects_size_budget` | `n_tokens <= size + 32` |
| `test_chunks_have_overlap` | Хвост prev чанка в next |
| `test_chunk_ids_and_indices_are_sequential` | `d1::0`, `d1::1`, ... |
| `test_fixed_strategy_covers_all_tokens` | token0 и token199 в union чанков |
| `test_short_document_single_chunk` | Короткий текст → 1 чанк |
| `test_long_sentence_is_split` | Предложение > size → несколько чанков |

Использует реальный токенизатор `all-MiniLM-L6-v2`.

---

## tests/test_retriever.py

| Тест | Что проверяет |
|------|---------------|
| `test_relevant_document_ranked_first` | 3 документа → правильный top-1 |
| `test_retrieve_respects_k` | len(hits)==k, monotonic dense_score |
| `test_bm25_hybrid_finds_exact_term` | RRF + rare term ZX-9917 |

Интеграционный тест с реальным эмбеддером — проверяет end-to-end retrieval quality
на синтетическом корпусе из 3 документов.

---

## requirements.txt — полный разбор

### Core ML

```
torch>=2.1
transformers>=4.40
datasets>=2.18
```

- **PyTorch** — backend для transformers и sentence-transformers.
- **transformers** — AutoTokenizer, AutoModelForQuestionAnswering (reader).
- **datasets** — загрузка SQuAD v2.

### Retrieval

```
sentence-transformers>=2.7,<5
sentencepiece>=0.2
faiss-cpu>=1.8
rank-bm25>=0.2.2
```

| Пакет | Роль |
|-------|------|
| sentence-transformers | Embedder, CrossEncoder reranker |
| sentencepiece | Токенizer multilingual моделей |
| faiss-cpu | Vector index |
| rank-bm25 | BM25Okapi для hybrid |

Ограничение `sentence-transformers<5` — см. [07-embeddings.md](07-embeddings.md).

### Generative

```
llama-cpp-python>=0.3
```

CPU-инференс GGUF. Может потребовать сборку из исходников.

### Evaluation

```
evaluate>=0.4
pandas>=2.0
matplotlib>=3.7
```

- **evaluate** — официальная метрика squad_v2.
- **pandas** — CSV-отчёты.
- **matplotlib** — графики ablation, threshold curve (backend Agg).

### Ingestion

```
pypdf>=4.0
```

Извлечение текста из PDF.

### Interfaces

```
streamlit>=1.33
typer>=0.12
```

### Tests

```
pytest>=8.0
```

---

## Переменные окружения при импорте

```8:13:shkerebert/__init__.py
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
```

Устанавливаются при первом `import shkerebert` — до загрузки transformers.

---

## scripts/download_models.py

Не входит в requirements как пакет — использует `huggingface_hub` (транзитивная
зависимость transformers/datasets).

Функции:

- `fetch(repo)` — snapshot_download с ретраями, ignore onnx/openvino;
- `fetch_dataset()` — load_dataset squad_v2;
- проверка наличия Qwen GGUF в `models/`.

---

## Что не покрыто тестами

- Reader (BERT forward) — проверяется через eval oracle F1 ≈ 81.
- Generator (Qwen) — проверяется через eval_generative (медленно).
- Streamlit UI — ручное тестирование.
- PDF ingest — edge cases (сканы, колонки).

Это осознанный компромисс: тяжёлые модели — в eval, логика — в unit-тестах chunking/retrieval.
