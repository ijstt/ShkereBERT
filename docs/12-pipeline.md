# 12. Пайплайн, hybrid-режим и абстенция

Модуль `shkerebert/pipeline.py`.

## RAGPipeline

Центральный оркестратор: retrieval → reader → Answer + sources.

### Три режима answer()

| mode | Метод | Логика |
|------|-------|--------|
| `extractive` | `_answer_extractive` | BERT span + τ |
| `generative` | `_answer_generative` | Qwen + маркер отказа |
| `hybrid` | `_answer_hybrid` | Qwen + BERT-верификация |

## Уровень 1: retrieval-level абстенция

Общий для **всех** режимов:

```84:88:shkerebert/pipeline.py
        if not retrieved or retrieved[0].dense_score < self.cfg.retriever.min_score:
            return Answer(question, "", False, float("-inf"),
                          "retrieval below min_score", mode,
                          self._sources(retrieved, None))
```

`min_score=0.15` — эмпирический порог косинуса. Если лучший фрагмент семантически
далёк от вопроса, нет смысла вызывать ридер (экономия + меньше ложных ответов).

**Важно:** используется `dense_score`, не RRF/rerank score.

## Уровень 2a: extractive abstention

```97:107:shkerebert/pipeline.py
        spans = self.reader.read(question, [r.chunk for r in retrieved])
        best = spans[0]
        confidence = best.score - best.null_score
        tau = self.cfg.reader.no_answer_threshold
        if best.gap > tau or not best.text:
            return Answer(..., is_answerable=False, reason=f"reader no-answer (gap=...", ...)
        return Answer(..., is_answerable=True, answer=best.text, ...)
```

## Уровень 2b: generative abstention

```109:117:shkerebert/pipeline.py
        gen = self.generator.generate(question, [r.chunk for r in retrieved])
        if not gen.is_answerable:
            return Answer(..., is_answerable=False, reason="generator no-answer", ...)
        return Answer(..., answer=gen.answer, reason=f"generated (cites={gen.cited_indices})", ...)
```

## Hybrid-режим — анти-галлюцинация

```119:139:shkerebert/pipeline.py
    def _answer_hybrid(self, question, retrieved):
        gen = self.generator.generate(...)
        spans = self.reader.read(...)
        best = spans[0]
        extractive_supports = (best.gap <= tau) and bool(best.text)

        if not gen.is_answerable:
            if extractive_supports:
                return Answer(..., best.text, True, ..., "generator abstained; extractive fallback", ...)
            return Answer(..., is_answerable=False, ..., "both abstained", ...)

        verdict = "verified by extractive" if extractive_supports else "UNVERIFIED (возможна галлюцинация)"
        return Answer(..., gen.answer, True, ..., f"generated, {verdict}", ...)
```

**Логика:**

| Generator | Extractive | Результат |
|-----------|------------|-----------|
| abstain | confident span | **Fallback** на extractive |
| abstain | abstain | «нет ответа» |
| answer | span supports | answer + `verified by extractive` |
| answer | no support | answer + `UNVERIFIED` |

UNVERIFIED на impossible-вопросах — перехват ~70% галлюцинаций (eval_generative).

## Источники (цитирование)

```142:149:shkerebert/pipeline.py
    def _sources(self, retrieved, answer_chunk_id):
        return [Source(r.chunk.id, r.chunk.title, r.chunk.text, r.dense_score,
                       r.chunk.id == answer_chunk_id) for r in retrieved]

    def _sources_multi(self, retrieved, answer_chunk_ids):
        ids = set(answer_chunk_ids or [])
        return [Source(..., r.chunk.id in ids) for r in retrieved]
```

- Extractive: `is_answer_source=True` на чанке с лучшим span.
- Generative: на всех чанках из `cited_chunk_ids` (парсинг `[n]`).

## Answer.display

```44:48:shkerebert/pipeline.py
    @property
    def display(self) -> str:
        if not self.is_answerable:
            return "В документе нет ответа на этот вопрос."
        return self.answer
```

## Калибровка τ (eval)

Протокол в `eval/eval_e2e.py`:

1. Один прогон reader → сохранить `gap` и `text` для каждого вопроса.
2. Свип τ по сетке (31 точка между 2–98 перцентилями gap).
3. Для каждого τ: abstain если `gap > τ` OR `top1_score < min_score` OR пустой text.
4. Считать squad_v2 EM/F1.
5. Выбрать τ* = argmax F1 на **calibration**-сплите.
6. Отчитаться на **held-out test** с фиксированным τ*.

Это стандартный инференс SQuAD v2 — честные метрики без переобучения на test.

## Диаграмма решений

```
                    retrieve(question)
                           │
                    top1.dense_score < 0.15?
                      yes │ no
                          ▼
                    ┌─ extractive ─┐  generative   hybrid
                    │ gap > τ?     │  marker?      gen + verify
                    └──────────────┘
                          │
                    Answer + sources
```

## confidence

- **extractive:** `confidence = best.score - best.null_score` (= `-gap`).
- **generative/hybrid:** `float("nan")` — нет калиброванной вероятности.

CLI/UI показывают confidence только если не NaN:

```39:40:shkerebert/cli.py
    conf = "" if ans.confidence != ans.confidence else f"confidence={ans.confidence:.2f}; "
```

(`x != x` — проверка на NaN без math.isnan.)
