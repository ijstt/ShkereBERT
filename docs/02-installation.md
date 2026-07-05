# 02. Установка и запуск

## Требования

- Python 3.10+
- ~4 ГБ RAM для extractive-режима; ~8+ ГБ для generative (Qwen 3B Q4)
- Интернет — только для первой загрузки моделей и датасета SQuAD

## Виртуальное окружение

```bash
cd /path/to/ShkereBERT
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
export PYTHONPATH="$PWD"
```

Флаг `--system-site-packages` позволяет переиспользовать системный PyTorch, если он уже
установлен — экономит место и время.

При импорте пакета отключаются TF/Flax-бэкенды transformers (иначе на Keras 3 возможен
краш sentence-transformers):

```8:13:shkerebert/__init__.py
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
```

## Загрузка моделей

Скрипт `scripts/download_models.py` качает HF-модели с ретраями (устойчив к обрывам):

```bash
PYTHONPATH=. .venv/bin/python scripts/download_models.py            # обязательные
PYTHONPATH=. .venv/bin/python scripts/download_models.py --extras  # + reranker, roberta-base
```

| Модель | Назначение | Обязательна |
|--------|------------|-------------|
| `sentence-transformers/all-MiniLM-L6-v2` | Эмбеддер EN | да |
| `deepset/tinyroberta-squad2` | Extractive reader | да |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Эмбеддер EN+RU | да |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker | нет (`--extras`) |
| `deepset/roberta-base-squad2` | Сильнее reader | нет (`--extras`) |
| `rajpurkar/squad_v2` | Датасет для eval | да |

**Qwen GGUF** не качается скриптом — копируется вручную в `models/`:

```bash
cp /path/to/qwen2.5-3b-instruct-q4_k_m.gguf models/
```

См. `models/README.md`. Путь по умолчанию:

```55:57:shkerebert/config.py
class GeneratorConfig:
    model_path: str = "models/qwen2.5-3b-instruct-q4_k_m.gguf"
```

## Переменные окружения

| Переменная | Реализована | Эффект |
|------------|-------------|--------|
| `SHKEREBERT_DATA` | да | Каталог кэша эмбеддингов (default: `~/.cache/shkerebert`) |
| `SHKEREBERT_EMBED` | **нет в коде** | Упоминается в `run.sh`, но `config.py` не читает |
| `SHKEREBERT_LLM` | **нет в коде** | Для смены пути к GGUF — править `GeneratorConfig.model_path` |

Для многоязычного поиска используйте `multilingual_config()` из `config.py` или
`with_overrides(cfg, retriever__embed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")`.

## CLI (Typer)

Модуль: `shkerebert/cli.py`.

```bash
# Интерактивный чат на SQuAD (200 контекстов)
.venv/bin/python -m shkerebert.cli chat --squad-n 300

# Чат по своему файлу
.venv/bin/python -m shkerebert.cli chat --file demo/machine_learning.txt

# Один вопрос
.venv/bin/python -m shkerebert.cli ask --file doc.pdf "What is ...?"

# Generative / hybrid
.venv/bin/python -m shkerebert.cli chat --file doc.txt --mode generative
.venv/bin/python -m shkerebert.cli chat --file doc.txt --mode hybrid
```

Команда `chat` в цикле вызывает `pipe.answer(q, mode=mode)` и печатает ответ + источники.

## Streamlit UI

```bash
.venv/bin/streamlit run app/streamlit_app.py
# или
./run.sh web
```

UI (`app/streamlit_app.py`):

- sidebar: выбор SQuAD или загрузка PDF/TXT;
- выбор режима ридера (`extractive` / `generative` / `hybrid`);
- `@st.cache_resource` кэширует построенный пайплайн;
- показывает ответ, confidence, reason и expander'ы с текстом источников.

## Обёртка run.sh

```bash
./run.sh web              # Streamlit
./run.sh chat [FILE]      # extractive CLI
./run.sh gen  [FILE]      # generative CLI
./run.sh ru   [FILE]      # generative + RU-демо (bank_products_ru.txt)
./run.sh ask "QUESTION" [FILE] [MODE]
./run.sh test             # pytest
./run.sh eval             # retrieval + e2e
./run.sh eval-extra       # longdoc, multiseed, variants, generative
./run.sh check            # проверка venv и моделей
```

**Замечание:** `./run.sh ru` экспортирует `SHKEREBERT_EMBED`, но код конфига это не
подхватывает — для реального RU-поиска нужно передать `multilingual_config()` в
`RAGPipeline.build(...)`.

## Первый запуск: что происходит

1. Загрузка документов (`ingest.py`).
2. Чанкинг всех документов (`chunking.py`).
3. Эмбеддинг чанков — **самый долгий шаг**; результат кэшируется на диск
   (`embeddings.py`).
4. Построение FAISS-индекса в памяти (`index.py`).
5. При первом generative-запросе — загрузка Qwen GGUF (~2 ГБ, несколько секунд).

Повторные запуски с тем же корпусом и параметрами чанкинга используют кэш эмбеддингов.

## Типичные проблемы

| Симптом | Причина | Решение |
|---------|---------|---------|
| `No module named shkerebert` | Нет PYTHONPATH | `export PYTHONPATH=$PWD` |
| Долгая загрузка HF | Нет локального кэша | `scripts/download_models.py` |
| `generative` не стартует | Нет GGUF | Скопировать в `models/` |
| OOM на CPU | Qwen 7B на слабой машине | Использовать Q4 3B или extractive |
| Низкое качество на RU TXT | EN-эмбеддер | `multilingual_config()` |
