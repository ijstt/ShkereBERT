# Быстрый старт — запуск ShkereBERT за 3 команды

> **Требования:** Linux/macOS/WSL, Python 3.10+, 8+ GB RAM, **CPU-only** (GPU не нужен).  
> Модели скачиваются автоматически при первом запуске (~1.5 GB).

---

## 1. Установка окружения

```bash
# 1. Клонируем репозиторий
git clone <repo-url>
cd shkerebert

# 2. Виртуальное окружение
python -m venv .venv
source .venv/bin/activate

# 3. Зависимости
pip install -U pip
pip install -r requirements.txt
```

> **requirements.txt** закрепляет версии: `faiss-cpu==1.8.0`, `sentence-transformers==3.0.1`, `llama-cpp-python==0.2.90`, `rank-bm25==0.2.2`, `transformers==4.41.2`, `streamlit==1.35.0`, `typer==0.12.3`, `evaluate==0.4.1`, `datasets==2.19.1`.

---

## 2. Скачивание моделей (один раз)

```bash
# Вариант А — через скрипт (с ретраями/бэкоффом)
python scripts/download_models.py

# Вариант Б — ручной префетч (если скрипт не сработал)
# Эмбеддеры (HF cache)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Extractive reader (HF cache)
python -c "from transformers import AutoModelForQuestionAnswering, AutoTokenizer; AutoModelForQuestionAnswering.from_pretrained('deepset/tinyroberta-squad2'); AutoTokenizer.from_pretrained('deepset/tinyroberta-squad2')"

# Generative LLM (GGUF → models/)
# Qwen 2.5 3B Instruct Q4_K_M (~2 GB)
wget -P models/ https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
```

**Структура `models/` после скачивания:**
```
models/
├── qwen2.5-3b-instruct-q4_k_m.gguf
└── README.md          # лицензии, SHA256, источники
```

---

## 3. Запуск через единый скрипт `run.sh`

```bash
# Сделать исполняемым (один раз)
chmod +x run.sh

# Режимы:
./run.sh streamlit   # Веб-UI (порт 8501) — рекомендуется для демо
./run.sh cli         # Интерактивный чат в терминале
./run.sh ask "Вопрос" [--file путь/к/файлу.txt] [--mode extractive|generative|hybrid] [--k 5]
./run.sh eval        # Полный прогон оценки (SQuAD + LongDoc + ablations) → eval/results/
./run.sh download    # Только скачать модели
```

**Переменные окружения (опционально):**
```bash
export SHKEREBERT_CACHE_DIR=~/.cache/shkerebert   # куда класть эмбеддинги/индексы
export SHKEREBERT_MODELS_DIR=./models             # где лежат GGUF
export OMP_NUM_THREADS=4                          # потоки FAISS/BLAS
```

---

## 4. Streamlit Web UI (режим `streamlit`)

```bash
./run.sh streamlit
# Откроется http://localhost:8501
```

**UI элементы:**
| Элемент | Назначение |
|---------|------------|
| **Источник знаний** | `SQuAD (demo)` — встроенные 1500 вопросов / `Загрузить файл` — свой `.txt/.pdf` |
| **Режим ответа** | `Extractive` (BERT, быстро, с отказом) / `Generative` (Qwen, подробно) / `Hybrid` (факты + объяснение) |
| **Top-K чанков** | 1–20 (по умолчанию 5) |
| **История** | Последние 10 Q/A с раскрывающимися источниками |
| **Кнопка «Пересобрать индекс»** | При смене файла или параметров чанкинга |

**Кэширование (в `app/streamlit_app.py`):**
- `@st.cache_resource` на `build_pipeline_squad` / `build_pipeline_file` — пайплайн не пересобирается при перерисовке UI.
- Индекс инвалидируется автоматически через `_corpus_signature` (хеш чанков + конфиг эмбеддера).

---

## 5. CLI (режим `cli`)

```bash
# Интерактивный REPL
./run.sh cli
# Команды внутри REPL:
#   /mode extractive|generative|hybrid
#   /k 10
#   /source squad|file <path>
#   /rebuild
#   /help
#   /quit

# Одиночный вопрос (для скриптов/CI)
./run.sh ask "Какая комиссия за перевод?" --mode hybrid --k 5
./run.sh ask "What is the capital of France?" --file demo/machine_learning.txt --mode extractive
```

**Ключевые функции в `shkerebert/cli.py`:**
- `chat()` — REPL-цикл с `typer` + `prompt_toolkit` (история, автодополнение).
- `ask()` — single-shot, печатает `Answer.display` (цветной вывод: ответ, уверенность, источники `[1]`, `[2]`).
- `_build(source, file_path)` — собирает `RAGPipeline` под выбранный корпус.
- `_print_answer(ans)` — форматированный вывод с latency и confidence.

---

## 6. Программное использование (Python API)

```python
from shkerebert import Config, RAGPipeline
from shkerebert.config import default_config

# 1. Конфиг (можно переопределить любые поля)
cfg = default_config()
cfg.chunk.size = 256
cfg.chunk.overlap = 64
cfg.retriever.top_k = 5
cfg.retriever.use_bm25 = True
cfg.retriever.use_reranker = False

# 2. Сборка пайплайна на SQuAD (demo)
pipe = RAGPipeline.build(cfg, squad_n=200)   # squad_n=None → весь validation

# 3. Ответ
ans = pipe.answer("What is the revenue of Apple?", mode="extractive", k=5)
print(ans.display)          # красивый вывод
print(ans.text)             # чистый текст ответа
print(ans.confidence)       # 0.0–1.0
print([s.text for s in ans.sources])  # тексты чанков-источников
```

**Режимы `mode`:**
| Режим | Класс метода | Когда использовать |
|-------|--------------|-------------------|
| `extractive` | `_answer_extractive` | Нужен точный фрагмент из документа, быстрый отказ, аудит |
| `generative` | `_answer_generative` | Свободный ответ, суммаризация, RU-язык |
| `hybrid` | `_answer_hybrid` | Extractive-факт + generative-объяснение, верификация галлюцинаций |

---

## 7. Оценка (режим `eval`)

```bash
# Полный прогон (5–10 мин на CPU)
./run.sh eval

# Результаты в eval/results/:
#   retrieval_*.csv          # Recall@k, MRR по вариантам ретривера
#   e2e_*.csv                # F1, EM, HasAns/NoAns на calibration/test
#   generative_*.csv         # Gold-containment, refusal, UNVERIFIED
#   longdoc_*.csv            # Влияние chunk size/overlap
#   ablation_*.csv           # Top-k, BM25, Reranker
#   multiseed_*.csv          # CI 95% по 5 сидам
#   *.png                    # Графики для слайдов
```

**Отдельные скрипты (для CI/экспериментов):**
```bash
python -m eval.eval_retrieval          # только retrieval
python -m eval.eval_e2e                # только end-to-end (calibration/test)
python -m eval.eval_generative         # только generative/hybrid
python -m eval.eval_longdoc            # только LongDoc
python -m eval.ablations               # ablation study
python -m eval.eval_multiseed          # multi-seed CI
```

---

## 8. Типичные проблемы и решения

| Симптом | Причина | Решение |
|---------|---------|---------|
| `ModuleNotFoundError: faiss` | Не установлен `faiss-cpu` | `pip install faiss-cpu==1.8.0` |
| `RuntimeError: model not found` | Модели не скачаны | `./run.sh download` или `python scripts/download_models.py` |
| `gguf: failed to load model` | Не тот путь к `.gguf` | Проверьте `SHKEREBERT_MODELS_DIR` и наличие файла в `models/` |
| `CUDA out of memory` | `llama-cpp-python` собран с CUDA | Пересоберите: `CMAKE_ARGS="-DGGML_CUDA=OFF" pip install --force-reinstall llama-cpp-python` |
| Долгий cold-start (~30 с) | Первая загрузка эмбеддера/ридера | Нормально; последующие запуски ~2 с за счёт кэша |
| `IndexFlatIP: dimension mismatch` | Эмбеддер сменился, старый индекс в кэше | Удалите `~/.cache/shkerebert/` или нажмите «Пересобрать индекс» в UI |

---

## 9. Полезные ссылки в коде

| Задача | Файл / Функция |
|--------|----------------|
| Точка входа CLI | `shkerebert/cli.py` → `app()` (typer) |
| Точка входа Streamlit | `app/streamlit_app.py` → `main()` |
| Сборка пайплайна | `shkerebert/pipeline.py` → `RAGPipeline.build()` |
| Скачивание моделей | `scripts/download_models.py` → `fetch()`, `fetch_dataset()` |
| Конфигурация по умолчанию | `shkerebert/config.py` → `default_config()` |
| Запуск оценки | `eval/__init__.py` → `run_all()` |

---

> **Совет для куратора:** для демо используйте `./run.sh streamlit`, выберите источник «SQuAD (demo)», режим «Extractive» — вопросы из SQuAD v2 отвечаются за ~300 мс с цитированием чанков.
