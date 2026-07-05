# Гибридный поиск: Dense + BM25 (RRF)

## Мотивация

| Тип запроса | Dense (MiniLM) | BM25 |
|---|---|---|
| Перефразировка («когда появился…» ≈ «в каком году введена…») | ✅ Силен | ❌ Слаб |
| Точные термины (номера тарифов, ИНН, артикулы, коды) | ❌ «Плывёт» | ✅ Силен |
| Короткие вопросы (1–2 слова) | ❌ Недостаточно контекста | ✅ Хорошо |

**Вывод**: нужен гибрид. Dense — базовый, BM25 — усилитель для точных совпадений.

## Reciprocal Rank Fusion (RRF)

Скоры dense (cosine ∈ [0,1]) и BM25 (TF-IDF-like, несравнимые шкалы) **нельзя складывать
напрямую**. RRF фьюзит **ранги**, а не скоры:

```
RRF(d) = Σ_{systems} 1 / (k + rank_system(d) + 1)
```

Где `k = 60` (эмпирическая константа, стандарт в IR), `rank` — 0-based позиция в
ранжированном списке каждой системы.

### Реализация (retriever.py)

```python
def _fuse_bm25(self, question: str, qvec: np.ndarray,
               dense_cands: list[RetrievedChunk]) -> list[RetrievedChunk]:
    # 1. BM25 скоры для всего корпуса
    bm_scores = self.bm25.get_scores(_bm25_tokens(question))
    bm_top = top_indices(bm_scores, self.cfg.retriever.bm25_pool)

    # 2. RRF аккумулятор
    rrf = defaultdict(float)
    dense_scores = {}  # сохраняем dense_score для абстенции

    for rank, c in enumerate(dense_cands):
        i = self._pos[c.chunk.id]
        rrf[i] += 1.0 / (rrf_k + rank + 1)
        dense_scores[i] = c.dense_score

    for rank, i in enumerate(bm_top):
        rrf[i] += 1.0 / (rrf_k + rank + 1)
        if i not in dense_scores:
            # Кандидат только из BM25 — восстанавливаем dense вектор из FAISS
            dense_scores[i] = float(np.dot(qvec, self.index.index.reconstruct(i)))

    # 3. Сортировка по RRF, возврат RetrievedChunk(score=RRF, dense_score=cosine)
    order = sorted(rrf, key=rrf.get, reverse=True)
    return [RetrievedChunk(chunks[i], rrf[i], dense_scores[i]) for i in order]
```

### Важные детали

1. **dense_score сохраняется всегда** — на нём живёт retrieval-level абстенция
   (`min_score=0.15` в конфиге). RRF-скор неинтерпретируем как вероятность.
2. **Пул кандидатов** — берём `max(top_k, bm25_pool, rerank_pool)` из FAISS, затем
   фьюзим/переоцениваем, режем до `top_k`.
3. **BM25 строится лениво** — при первом гибридном запросе (`rank_bm25.BM25Okapi`).

## Результаты (eval/results/retrieval_variants.csv, n=400 test)

| Вариант | Recall@5 | MRR | F1 (e2e) | Поиск мс/вопрос |
|---|---|---|---|---|
| dense | 0.955 | 0.871 | 80.11 | 7.6 |
| **dense+BM25 (RRF)** | **0.972** | **0.890** | 80.11 | 8.5 |

- **Retrieval метрики выросли** (+1.7 п. recall, +2 п. MRR) за +0.9 мс — гибрид почти бесплатен.
- **Но e2e F1 не сдвинулся** — на SQuAD (перефразированные вопросы) dense уже силен,
  а добытые BM25 чанки редко становятся источником ответа.
- **Где гибрид окупится**: доменные документы с точными терминами/номерами (тарифы, ИНН,
  артикулы). Юнит-тест `test_bm25_hybrid_finds_exact_term` демонстрирует этот случай.

## Конфигурация

```python
RetrieverConfig(
    use_bm25=True,
    bm25_pool=50,      # сколько кандидатов из BM25 для фьюза
    rrf_k=60,          # константа RRF
)
```
