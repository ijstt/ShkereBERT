# 16. Экспериментальные результаты и выводы

Сводка зафиксированных эксперimentов. Сырые CSV/PNG — `eval/results/`.
Хронология решений — `CHANGELOG.md`.

---

## Retrieval (baseline, n≈1000)

Корпус ~582 документа, ~377 answerable-вопросов.

| Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|----------|----------|----------|-----------|-----|
| 0.788 | 0.920 | **0.942** | 0.971 | 0.859 |

**Вывод:** Recall@5 = 94% → `top_k=5` обоснован.

---

## End-to-end QA (chunk=256, k=5, extractive)

Протокол: τ* на calibration, метрики на held-out test.

| setup | τ* | EM | F1 | HasAns F1 | NoAns F1 | latency |
|-------|-----|-----|-----|-----------|----------|---------|
| **retrieval (пайплайн)** | −6.3 | 73.6 | **75.8** | 66.2 | 84.6 | 229 мс/q |
| oracle (gold context) | −4.4 | 79.1 | **81.0** | 76.9 | 84.9 | 47 мс/q |

**Retrieval loss = 81.0 − 75.8 = 5.2 F1** — измеренная цена этапа поиска.

Oracle F1 ≈ published dev-F1 tinyroberta-squad2 → reader реализован корректно.

График калибровки: `eval/results/e2e_threshold_curve.png`.

---

## Ablation: chunk_size × top_k (n=400–500)

### chunk_size (k=3–10, best F1)

| chunk | Recall@1 | best F1 | вывод |
|-------|----------|---------|-------|
| 128 | **0.854** | **78.1** (k=3) | лучший recall и F1, больше чанков |
| **256** | 0.829 | 77.5 (k=10) | **baseline** — баланс |
| 512 | 0.824 | 77.3 | крупные чанки шумят |

### top_k

- F1 насыщается к k=3–5;
- Latency reader линейна: k=1 ≈ 50 мс → k=10 ≈ 550 мс;
- **k=5** — оптимум качество/скорость.

На **коротких** SQuAD-контекстах разница chunk_size мала (медиана ~141 токен).

---

## Long-document experiment (eval_longdoc)

Склейка статей Wikipedia → медиана ~5.3k токенов, 100% документов режутся.

| chunk | F1 (test) | NoAns F1 | latency |
|-------|-----------|----------|---------|
| 64 | **74.7** | **88.4** | 181 мс |
| 256 | 69.3 | 79.3 | 371 мс |

**Механизм:** HasAns_F1 ~56–58 у всех размеров; разница в NoAns_F1 — меньше шума в
контексте → чище null-score решение reader'а.

**answer-recall@5** ~85–88% vs **doc-recall@5** ~95–98% — статья находится, но нужный
абзац внутри неё — сложнее.

---

## Multi-seed CI (5 seeds, n=800)

F1 = **76.9 ± 3.9** (95% CI). Baseline 75.8 внутри интервала.

Разница конфигураций в 1–2 F1 на малых выборках — шум; доверять эффектам > CI.

τ* по сидам: от −9.3 до −4.4 — **порог нужно калибровать**, не хардкодить навсегда.

---

## Retrieval variants (BM25, reranker)

| variant | recall@5 | MRR | e2e F1 |
|---------|----------|-----|--------|
| dense | 0.955 | 0.871 | 80.11 |
| dense+bm25 | **0.972** | **0.890** | 80.11 |

BM25: +1.7% recall, +0.9 мс retrieval — e2e F1 не изменился на SQuAD (перефразированные
вопросы). Ожидаемый выигрыш — доменные доки с точными терминами.

---

## Generative vs extractive vs hybrid (n=150, seed=123)

| mode | F1 | gold_containment | latency mean | abstain (impossible) |
|------|-----|------------------|--------------|---------------------|
| extractive | 76.0 | 0.93 | 0.28 с | 0.82 |
| generative | 43.8* | 0.89 | 10.4 с | 0.42 |
| hybrid | ~generative | ~0.89 | ~10.5 с | UNVERIFIED 70% на halluc |

\* strict F1 занижен многословием LLM; containment 0.89 — адекватнее для generative.

**Hybrid:** перехватывает ~70% галлюцинаций на impossible через UNVERIFIED-метку.

---

## Ключевые инженерные выводы

1. **Двухуровневая абстенция** (retrieval min_score + reader τ/marker) — NoAns F1 84.6.
2. **Калибровка τ** — часть пайплайна, не post-hoc; свип без перезапуска модели.
3. **Кэш эмбеддингов** — критичен для итераций; ablation по top_k дешёвый.
4. **Chunk size** — зависит от длины документов; на PDF нужны меньшие чанки.
5. **Generative** — для UX и RU; **extractive** — для метрик и скорости.
6. **Hybrid** — guardrail против галлюцинаций, не замена extractive на SQuAD.

---

## Известные ограничения (повтор)

- Extractive — один span, EN-модель.
- τ и min_score — откалиброваны на SQuAD dev.
- PDF без OCR.
- `run.sh ru` не переключает embedder без правки кода (см. [02-installation.md](02-installation.md)).
- top-k не собирает ответ из далёких фрагментов.

---

## Где смотреть артеfacts

| Файл | Содержание |
|------|------------|
| `eval/results/retrieval.csv` | Recall@k, MRR |
| `eval/results/e2e_summary.csv` | retrieval vs oracle |
| `eval/results/e2e_threshold_curve.csv/png` | F1 vs τ |
| `eval/results/ablations.csv` | chunk × top_k |
| `eval/results/longdoc_chunking.csv/png` | длинные документы |
| `eval/results/multiseed.csv` | CI по seeds |
| `eval/results/retrieval_variants.csv` | BM25/reranker |
| `eval/results/generative_eval.csv` | режимы ридера |
