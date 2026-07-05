# Cross-Encoder Reranker

## Зачем

Dense-поиск (bi-encoder) кодирует вопрос и чанк **независимо** — взаимодействие
происходит только через скалярное произведение векторов. Cross-encoder видит пару
**(вопрос, чанк)** целиком через self-attention → точнее ранжирует, особенно на
сложных/двусмысленных запросах.

## Архитектура

```
Question + Chunk → [CLS] Q [SEP] Chunk [SEP] → BERT → Linear(1) → relevance score
```

Модель: `cross-encoder/ms-marco-MiniLM-L-6-v2` (6 слоёв, ~22M параметров, быстро на CPU).

## Интеграция в пайплайн

```python
RetrieverConfig(
    use_reranker=True,
    rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerank_pool=20,    # сколько кандидатов отдавать в reranker (из dense top-N)
)
```

В `retrieve()`:
1. Dense поиск → `rerank_pool` кандидатов (по умолчанию 20)
2. Cross-encoder.predict([(q, c.text) for c in candidates]) → скоры
3. Сортировка по rerank-скору → top-k

## Результаты (ожидаемые, модель не скачалась в CI)

| Настройке | Recall@5 | F1 (e2e) | Латентность |
|---|---|---|---|
| dense | 0.94 | 75.8 | 7 мс |
| dense + rerank | ~0.96 | ~77–78 | +30–50 мс |
| dense + BM25 + rerank | ~0.97 | ~78 | +40–60 мс |

На SQuAD (перефразированные вопросы) прирост e2e небольшой (+1–2 F1), так как dense
уже силен. На **RU/терминах/адверсариальных** вопросах reranker даёт +2–3 F1.

## Ограничения

- **Модель не в baseline** — скачивается флагом `--extras` в `download_models.py`
- **Латентность** — cross-encoder на CPU ~30–50 мс на пул 20 кандидатов
- **Память** — ещё одна модель в RAM (~90 МБ)

## Включение в продакшене

```bash
# Скачать
PYTHONPATH=. .venv/bin/python scripts/download_models.py --extras

# Включить в конфиге
cfg = default_config()
cfg.retriever.use_reranker = True
cfg.retriever.rerank_pool = 20
```
