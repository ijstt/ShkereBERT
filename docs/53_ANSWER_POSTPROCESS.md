# Постобработка ответа и логика абстенции

## Двухуровневая абстенция

```
Вопрос
  │
  ▼
┌─────────────────────────────────────┐
│  Retrieval-level абстенция          │
│  best_dense_score < min_score (0.15)│
└──────────────┬──────────────────────┘
               │ НЕТ
               ▼
┌─────────────────────────────────────┐
│  Reader-level абстенция             │
│  Extractive: gap > τ                │
│  Generative: маркер отказа в ответе │
│  Hybrid: оба + верификация          │
└──────────────┬──────────────────────┘
               │
               ▼
        Ответ / «Нет ответа»
```

## Уровень 1: Retrieval (pipeline.py)

```python
if not retrieved or retrieved[0].dense_score < self.cfg.retriever.min_score:
    return Answer(question, "", False, float("-inf"),
                  "retrieval below min_score", mode, sources)
```

- `min_score=0.15` (cosine similarity) — эмпирически: ниже этого вопроса либо не по
  документу, либо чанки случайны.
- Срабатывает **до запуска ридера** — экономит время.
- `dense_score` — чистый cosine, не RRF/rerank скор (шкала интерпретируема).

## Уровень 2: Extractive Reader (reader.py + pipeline.py)

```python
# В _answer_extractive:
best = spans[0]
confidence = best.score - best.null_score  # = -gap
tau = self.cfg.reader.no_answer_threshold  # τ ≈ -6.3 (калибровано)
if best.gap > tau or not best.text:
    return Answer(..., False, confidence,
                  f"reader no-answer (gap={best.gap:.2f} > tau={tau:.2f})", ...)
return Answer(..., True, confidence, "answer extracted", ...)
```

### Калибровка τ (eval/eval_e2e.py)

1. **Calibration split** (50% вопросов) — перебор τ по сетке, максимизация F1
2. **Test split** (остальные 50%) — отчёт при фиксированном τ*
3. **График** `e2e_threshold_curve.png` — F1 vs τ для HasAns/NoAns/Overall

**Результат**: τ* ≈ −6.3 (отрицательный! — модель завышает null-score).
На длинных доках τ* гуляет от −9.6 (chunk=64) до −1.7 (chunk=512).

## Уровень 2: Generative Reader (generator.py)

```python
# В generate():
markers = {marker, self.cfg.no_answer_marker, _EN_MARKER}
if any(m.upper().replace("Ё", "Е") in norm for m in markers):
    return GenAnswer("", False, raw=text, latency_s=dt)
```

- Маркер отказа: RU «НЕТ ОТВЕТА», EN «NO ANSWER» (принимаем оба варианта)
- Temperature=0 → детерминированно
- **Качество отказа слабее** (0.42 против 0.82 у extractive) — промпт не калибруется

## Уровень 2: Hybrid (pipeline.py)

```python
def _answer_hybrid(self, question, retrieved):
    gen = self.generator.generate(question, chunks)
    spans = self.reader.read(question, chunks)
    best = spans[0]
    extractive_supports = (best.gap <= tau) and bool(best.text)

    if not gen.is_answerable:
        if extractive_supports:
            return Answer(..., best.text, True, ..., "generator abstained; extractive fallback")
        return Answer(..., "", False, ..., "both abstained")

    verdict = "verified by extractive" if extractive_supports else "UNVERIFIED"
    return Answer(..., gen.answer, True, ..., f"generated, {verdict}")
```

### Логика верификации

| Генератор | Extractive | Результат |
|---|---|---|
| Ответ | Поддерживает (gap ≤ τ) | **Verified** — факт подтверждён |
| Ответ | Не поддерживает (gap > τ) | **UNVERIFIED** — возможна галлюцинация |
| Отказ | Поддерживает | **Fallback на extractive** — BERT нашёл, Qwen промахнулся |
| Отказ | Не поддерживает | **Both abstained** — честный отказ |

**Метрика**: на impossible-вопросах UNVERIFIED rate = 0.70 (перехватывает 70% галлюцинаций).

## Confidence в ответе

| Режим | Confidence | Интерпретация |
|---|---|---|
| Extractive | `best.score - best.null_score` (= -gap) | Чем больше, тем увереннее в ответе |
| Generative | `NaN` | Нет калибруемой уверенности |
| Hybrid | `NaN` | Используйте verdict (verified/UNVERIFIED) |

## Reason (причина решения)

Каждый `Answer` содержит `reason` — человекочитаемое объяснение для отладки/аудита:
- `"answer extracted"`
- `"reader no-answer (gap=-4.2 > tau=-6.3)"`
- `"retrieval below min_score"`
- `"generated, verified by extractive"`
- `"generated, UNVERIFIED (возможна галлюцинация)"`
- `"generator abstained; extractive fallback"`
- `"both abstained"`

В UI/CLI выводится рядом с ответом.
