# ShkereBERT — RAG-чат-бот по документу

Учебный проект: чат-бот, который отвечает на вопросы по текстовому документу(ам) через
**RAG-пайплайн** (retrieval + экстрактивный BERT-reader) и **умеет говорить «ответа нет»**.
Валидация — на [SQuAD v2](https://huggingface.co/datasets/rajpurkar/squad_v2) по официальным
метрикам **Exact Match / F1**.

## Идея и архитектура

```
Документы (SQuAD / PDF / TXT)
   → чанкинг (по предложениям + overlap, размер в токенах)
   → эмбеддинги (all-MiniLM-L6-v2, CPU, кэш)
   → индекс FAISS (cosine)
   → retrieval top-k
   → ридер (на выбор):
        • extractive — BERT tinyroberta-squad2 (span + null-score, порог τ)
        • generative — Qwen 2.5 3B локально (llama-cpp), grounded-ответ с цитатами
        • hybrid     — Qwen отвечает, BERT проверяет обоснованность (анти-галлюцинация)
   → абстенция (retrieval-level + reader-level)
   → ответ + ЦИТАТА фрагмента(ов)
```

**Режимы ридера** переключаются через `cfg.reader_mode` или аргумент `mode=` в
`RAGPipeline.answer(...)`. Генеративный ридер полностью офлайн (Qwen GGUF на CPU) — ни
один документ не покидает машину, что важно для on-premise-сценариев (напр. банк).

**Обработка «нет ответа»** двухуровневая: (1) если лучший фрагмент слишком непохож на
вопрос; (2) если reader считает `null_score - best_span_score > τ`. Порог τ калибруется на
dev-сплите по максимуму F1.

Подробное обоснование решений — в `shkerebert/*.py` (докстринги) и плане проекта.

## Установка

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
```

## Использование

CLI-чат:
```bash
.venv/bin/python -m shkerebert.cli chat --squad-n 300          # демо на SQuAD
.venv/bin/python -m shkerebert.cli chat --file path/to/doc.pdf # свой документ
.venv/bin/python -m shkerebert.cli ask --file doc.txt "What is ...?"
```

Веб-интерфейс:
```bash
.venv/bin/streamlit run app/streamlit_app.py
```

## Оценка (доказательство эффективности)

```bash
.venv/bin/python -m eval.eval_retrieval --n 1000   # Recall@k, MRR
.venv/bin/python -m eval.eval_e2e --n 1000         # EM/F1 + калибровка τ (+ график)
.venv/bin/python -m eval.ablations --n 500         # свипы chunk_size / top_k (+графики)
```

Результаты (CSV + PNG) складываются в `eval/results/`.

### Метрики (dev SQuAD v2, CPU-ноутбук)

**Retrieval** (корпус 582 документа, 377 answerable-вопросов):

| Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| 0.788 | 0.920 | **0.942** | 0.971 | 0.859 |

Recall@5 = 94% при 5 фрагментах — обоснование выбора `top_k=5`.

**End-to-end QA** (chunk=256, k=5, reader=tinyroberta-squad2). Порог τ подобран на
**calibration**-сплите, метрики — на **held-out test** (без утечки):

| setup | tau* | EM | F1 | HasAns F1 | NoAns F1 | latency |
|---|---|---|---|---|---|---|
| **retrieval (наш пайплайн)** | −6.3 | 73.6 | **75.8** | 66.2 | 84.6 | 229 мс/q |
| oracle-контекст (потолок reader'а) | −4.4 | 79.1 | **81.0** | 76.9 | 84.9 | 47 мс/q |

**Потеря на этапе retrieval = 5.26 F1** (измерено). Oracle F1=81.0 совпадает с эталонным
dev-F1 tinyroberta-squad2 → реализация reader'а корректна. Калибровка порога (на calibration):
`eval/results/e2e_threshold_curve.png`.

**Ablation** (`eval/results/ablations.csv`, `ablation_*.png`, n=400):

| chunk | Recall@1 | best F1 | вывод |
|---|---|---|---|
| 128 | **0.854** | **78.1** (k=3) | лучший recall@1 и F1, но больше чанков |
| 256 | 0.829 | 77.5 (k=10) | баланс качество/память |
| 512 | 0.824 | 77.3 | чуть хуже, крупные чанки шумят |

- **top_k**: F1 насыщается уже к k=3–5, а латентность растёт линейно (k=1 ≈ 50 мс →
  k=10 ≈ 550 мс, т.к. reader читает каждый фрагмент). Вывод: **k=5 — оптимум**
  (Recall@5 ≈ 0.95, F1 около пика, умеренная задержка).
- **chunk_size**: мелкие чанки (128–256) точнее для поиска; 512 начинает шуметь.
  Baseline фиксирует chunk=256, k=5 как компромисс качество/ресурсы.

## Тесты

```bash
.venv/bin/python -m pytest -q
```

## Ограничения

- **Экстрактивный reader** возвращает единый непрерывный span, поэтому не умеет
  «собирать» составной ответ. Пример: на вопрос «кто и в каком году ввёл терм"» из
  demo-документа система вернёт только год («1959»), а не имя+год.
- **Домен**. Оценка и модели заточены под энциклопедический стиль SQuAD (Wikipedia).
  На узкоспециальных/разговорных документах качество ниже без дообучения.
- **Порог абстенции τ** откалиброван на dev SQuAD v2; для нового корпуса его стоит
  перекалибровать (кривая F1 vs τ строится тем же `eval_e2e`).
- **Извлечение из PDF** зависит от pypdf: сканы без текстового слоя и сложная вёрстка
  (таблицы, колонки) извлекаются плохо — нужен OCR.
- **Язык**. Базовая конфигурация англоязычная (SQuAD). Русский — как stretch:
  multilingual-эмбеддер + `xlm-roberta`-squad reader (не входит в baseline).
- **Ретривер видит только чанки**: если ответ распределён по нескольким далёким
  фрагментам, top-k может не собрать полный контекст.

## Структура

```
shkerebert/   # пакет: config, ingest, chunking, embeddings, index, retriever, reader, pipeline, cli
app/          # Streamlit UI
eval/         # build_corpus, eval_retrieval, eval_e2e, ablations, results/
tests/        # юнит-тесты
```
