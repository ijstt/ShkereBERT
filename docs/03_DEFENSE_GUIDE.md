# Сценарий защиты и пул вопросов куратора (ML-инженер банка)

> **Цель:** пройти 30-минутную защиту уверенно, показать глубину инженерной проработки и честность метрик.  
> **Аудитория:** куратор из отдела ML банка — технический, задаёт проbing questions.  
> **Формат:** 10 мин демо → 20 мин Q&A. Документ = шпаргалка для команды.

---

## 1. Демо-сценарий (10 минут)

| Шаг | Действие | Что показываем | Ключевое сообщение |
|-----|----------|----------------|-------------------|
| 1 | `./run.sh streamlit` | Открывается UI за 3 с (кэш) | **Production-ready UX**, не ноутбук |
| 2 | Источник: **SQuAD (demo)**, режим: **Extractive** | Вводим вопрос из SQuAD: *"What is the population of Tokyo?"* | Ответ за **~300 мс**, есть цитата `[1]`, confidence 0.92 |
| 3 | Вводим **unanswerable** вопрос: *"Who won the 2050 World Cup?"* | Система отвечает **"В документе нет ответа"** (NoAns), confidence 0.87 | **Калиброванный отказ** — критично для банка |
| 4 | Переключаем режим на **Hybrid**, тот же вопрос | Extractive-факт + Generative-объяснение, метка `UNVERIFIED` если генерация не подтвердилась | **Анти-галлюцинация** встроена в пайплайн |
| 5 | Источник: **Загрузить файл** → `demo/bank_products_ru.txt` | Вопрос на русском: *"Какая комиссия за перевод для юрлица?"* | **RU-пайплайн работает** (multilingual embedder + Qwen) |
| 6 | Показываем **History** панель | Все Q/A с раскрывающимися источниками | **Audit trail** из коробки |
| 7 | (Опционально) CLI: `./run.sh ask "..." --mode extractive` | Одиночный вопрос за 0.3 с в терминале | **Интеграция в скрипты/боты** |

**Plan B (если Streamlit глючит):** сразу CLI — функционал идентичен.

---

## 2. Пул ожидаемых вопросов куратора с ответами

> **Стратегия:** честно, с цифрами, со ссылками на код/артефакты.  
> Если не знаем — «Хороший вопрос, мы это не измеряли, но вот как можно проверить».

### 2.1. Архитектура и стек

| Вопрос | Краткий ответ | Где в коде / доках |
|--------|---------------|-------------------|
| **Почему FAISS IndexFlatIP, а не HNSW/IVF?** | Корпус ≤ 10k чанков → exact search быстрее ANN (нет build time, нет recall loss). На 100k+ перейдём на HNSW. | `shkerebert/index.py` (`VectorIndex`), `docs/32_VECTOR_INDEX.md` |
| **Почему cosine через inner product?** | Векторы L2-нормированы при эмбеддинге (`normalize_embeddings=True`) → IP = cosine. Быстрее, нет sqrt. | `shkerebert/embeddings.py` (`Embedder.encode`), `shkerebert/index.py` |
| **Зачем BM25, если есть dense?** | Dense теряет точные термины (номера тарифов, ИНН, артикулы). BM25 + RRF даёт +1.7 п. Recall@5 на exact-match. | `shkerebert/retriever.py` (`_fuse_bm25`), `docs/43_HYBRID_FUSION.md` |
| **Почему Cross-encoder reranker опционален?** | На SQuAD e2e F1 не вырос (перефразированные вопросы). На RU/терминах — даёт +2–3 п. Включаем флагом. | `shkerebert/retriever.py` (`_load_reranker`), `docs/44_RERANKER.md` |
| **Почему tinyroberta-squad2, а не ruBERT?** | SQuAD v2 обучена на EN; multilingual reader даёт -8 п. F1 на EN. Для RU — отдельный ридер или generative. | `shkerebert/reader.py` (`_load_qa`), `docs/51_READER_MODEL.md` |
| **Как работает doc_stride?** | Sliding window по токенам с перекрытием; агрегация логитов по max. Позволяет читать контексты > max_seq_len. | `shkerebert/reader.py` (`_read_one`), `docs/52_SPAN_EXTRACTION.md` |

### 2.2. Чанкинг и retrieval quality

| Вопрос | Краткий ответ | Где в коде / доках |
|--------|---------------|-------------------|
| **Почему chunk=256, overlap=64?** | Эмпирически на SQuAD (медиана контекста 141 токен) — разницы нет. На **LongDoc (5k токенов)** chunk=64 даёт **+5.4 п. F1** и **вдвое быстрее**. | `eval/eval_longdoc.py`, `docs/95_LONGDOC_EVAL.md` |
| **Как overlap влияет на recall?** | Overlap=64 (25%) гарантирует, что ответ не разорван на границе чанков. Тесты: `test_overlap_respected`. | `shkerebert/chunking.py` (`_chunk_sentence_aware`), `tests/test_chunking.py` |
| **Sentence-aware vs fixed-token — что выбрали?** | По умолчанию **sentence-aware** (сохраняет смысловые границы). Fallback на токены, если предложение > chunk_size. | `shkerebert/chunking.py` (`chunk_document`), `docs/23_CHUNKING_ALGORITHMS.md` |
| **Как обрабатываете таблицы/PDF-вёрстку?** | `pypdf` извлекает только текстовый слой. Таблицы теряются. Для пилота нужен `pdfplumber` / `unstructured` + table-to-text. | `shkerebert/ingest.py` (`load_pdf`), `docs/10_DATA_SOURCES.md` |

### 2.3. Extractive Reader и калибровка отказа (τ)

| Вопрос | Краткий ответ | Где в коде / доках |
|--------|---------------|-------------------|
| **Как работает «нет ответа»?** | BERT выдаёт `null_score` (logit [CLS]). Считаем `gap = null_score - best_span_score`. Если `gap > τ` → отказ. | `shkerebert/reader.py` (`_extract`), `docs/53_ANSWER_POSTPROCESS.md` |
| **Как подбирается τ?** | **Calibration split** (50% SQuAD val): максимизируем F1. **Test split** (остальные 50%) — честный отчёт. Раньше калибровали на тесте → F1 78.3 → 75.8 (утечка). Исправили. | `eval/eval_e2e.py` (`calibrate`, `score_at_tau`), `docs/93_E2E_EVAL.md` |
| **Какие метрики на held-out test?** | **F1 75.8**, EM 73.6, **NoAns F1 84.6**, HasAns F1 66.2. Oracle (gold context) F1 81.0 → retrieval loss 5.3 п. | `eval/results/e2e_test.csv`, `docs/ВЫВОДЫ_ПО_ДАННЫМ.md` |
| **Почему HasAns F1 ниже NoAns?** | Ридер консервативен (τ высокий) → пропускает некоторые has-ans. На банковских данных τ перекалибруем. | `eval/eval_e2e.py` (`score_at_tau`) |

### 2.4. Generative / Hybrid и галлюцинации

| Вопрос | Краткий ответ | Где в коде / доках |
|--------|---------------|-------------------|
| **Почему Qwen 2.5 3B Q4_K_M?** | 2 GB, работает на CPU 4 потоках ~10 с/ответ, хорошее качество RU/EN, лицензия Apache 2.0. | `shkerebert/generator.py` (`_load_llm`), `models/README.md` |
| **Как промпт заземляет генерацию?** | System prompt: «Отвечай ТОЛЬКО по контексту. Если нет — пиши "В контексте нет ответа"». Чанки нумеруются `[1]`, `[2]`. | `shkerebert/generator.py` (`_build_prompt`), `docs/62_PROMPT_BUILDING.md` |
| **Что такое UNVERIFIED в Hybrid?** | Extractive-ответ = факт. Generative — объяснение. BERT-ридер проверяет: содержит ли генерация extractive-спан? Если нет → метка `UNVERIFIED`. Ловит **70% галлюцинаций**. | `shkerebert/pipeline.py` (`_answer_hybrid`), `docs/75_HYBRID_MODE.md` |
| **Метрики generative на 150 вопросах?** | Gold-containment 0.89, Refusal на impossible 0.42, False refusal 0.08. Hybrid: UNVERIFIED rate 0.70. | `eval/eval_generative.py`, `eval/results/generative_test.csv` |

### 2.5. Оценка и воспроизводимость

| Вопрос | Краткий ответ | Где в коде / доках |
|--------|---------------|-------------------|
| **Как гарантируете отсутствие утечки?** | Calibration/Test split (фиксированные сиды), τ только на calibration. Oracle baseline измерен отдельно. Все сиды в конфигах. | `eval/eval_e2e.py`, `eval/ablations.py`, `docs/93_E2E_EVAL.md` |
| **Что такое multi-seed CI?** | 5 сидов (42, 123, 456, 789, 999) → прогоняем полный пайплайн → F1 = 76.9 ± 3.9 (95% CI). Разброс учитываем в выводах. | `eval/eval_multiseed.py`, `eval/results/multiseed_summary.csv` |
| **Можно ли воспроизвести за 1 команду?** | `./run.sh eval` → все CSV/PNG в `eval/results/`. Зависимости закреплены в `requirements.txt`. | `run.sh`, `eval/__init__.py` |
| **Есть ли дата-дрift мониторинг?** | Пока нет. Для пилота добавим логирование (вопрос, режим, confidence, sources) + периодическая перекалибровка τ. | `docs/100_CACHING.md` (планы) |

### 2.6. Прод и эксплуатация

| Вопрос | Краткий ответ | Где в коде / доках |
|--------|---------------|-------------------|
| **Cold-start latency?** | Первая загрузка эмбеддера/ридера/llm ~30 с (HF cache + llama.cpp