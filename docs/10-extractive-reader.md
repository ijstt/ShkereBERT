# 10. Экстрактивный ридер (BERT-span)

Модуль `shkerebert/reader.py`.

## Задача extractive QA

Модель находит **непрерывный span** текста в контексте, который отвечает на вопрос.
Не генерирует новые слова — только указывает start/end позиции.

Модель: `deepset/tinyroberta-squad2` — RoBERTa-tiny, дообученная на SQuAD v2.

## SQuAD v2 и «нет ответа»

SQuAD v1 — только answerable. **SQuAD v2** добавляет unanswerable вопросы.
Модель обучена выдавать **null answer** через специальные logits на `[CLS]`:

```
null_score      = start_logit[CLS] + end_logit[CLS]
best_span_score = max(start[i] + end[j])  по валидным i,j в контексте
gap             = null_score - best_span_score
```

Если `gap > 0` — модель склоняется к «ответа нет» для этого фрагмента.

## SpanAnswer

```25:37:shkerebert/reader.py
@dataclass
class SpanAnswer:
    text: str
    score: float          # best_span_score
    null_score: float
    start_char: int
    end_char: int
    chunk: Chunk

    @property
    def gap(self) -> float:
        return self.null_score - self.score
```

## Reader._extract — ядро инференса

```55:85:shkerebert/reader.py
    def _extract(self, start_logits, end_logits, offsets, seq_ids, chunk):
        null_score = float(start_logits[0] + end_logits[0])
        ctx_positions = np.where(np.array([sid == 1 for sid in seq_ids]))[0]
        ...
        n_best = 20
        starts = ctx_positions[np.argsort(start_logits[ctx_positions])[-n_best:]]
        ends = ctx_positions[np.argsort(end_logits[ctx_positions])[-n_best:]]
        for s in starts:
            for e in ends:
                if e < s or (e - s + 1) > cfg.max_answer_len:
                    continue
                score = float(start_logits[s] + end_logits[e])
                ...
        start_char, end_char = int(offsets[s][0]), int(offsets[e][1])
        text = chunk.text[start_char:end_char].strip()
```

**Детали:**

1. `sequence_ids`: 0 = question tokens, 1 = context tokens, None = special/padding.
2. Перебор top-20 start × top-20 end (не полный O(n²) — экономия на CPU).
3. `max_answer_len=30` — ограничение длины span в токенах.
4. `offset_mapping` — маппинг token → char position в исходном `chunk.text`.

## Токенизация пары question+context

```91:101:shkerebert/reader.py
        enc = self.tokenizer(
            question, chunk.text,
            truncation="only_second", max_length=cfg.max_seq_len,
            return_offsets_mapping=True, return_tensors="pt",
        )
        with torch.no_grad():
            out = self.model(**enc)
```

- `truncation="only_second"` — при переполнении режется **контекст**, вопрос целиком.
- `max_seq_len=384` — стандарт для SQuAD-моделей.

## read() — несколько чанков

```103:112:shkerebert/reader.py
    def read(self, question, chunks):
        answers = [self._read_one(question, ch) for ch in chunks]
        answers.sort(key=lambda a: a.score, reverse=True)
        return answers
```

Каждый чанк обрабатывается **отдельно** (без batch padding) — на CPU быстрее, чем
паддинг до max длины в батче (замерено в docstring).

Лучший span — первый после сортировки по `best_span_score` (не по gap).

## Порог абстенции τ

В pipeline:

```97:107:shkerebert/pipeline.py
        best = spans[0]
        confidence = best.score - best.null_score  # = -gap
        tau = self.cfg.reader.no_answer_threshold
        if best.gap > tau or not best.text:
            return Answer(..., is_answerable=False, ...)
```

Default τ = **-6.3** — подобран на calibration-сплите (`eval/eval_e2e.py`).

**Важно:** τ **не переносится** между конфигурациями (chunk size, выборка) — нужна
перекалибровка. На longdoc τ гуляет от -9.6 (chunk 64) до -1.7 (chunk 512).

## Oracle baseline

`eval/eval_e2e.py::collect_oracle_predictions` — reader на **золотом** контексте
без retrieval. Oracle F1 = **81.0** ≈ dev-F1 tinyroberta-squad2 → реализация корректна.

## Ограничения extractive

- Один непрерывный span — не «собирает» ответ из двух мест.
- Англоязычная модель — на RU без замены reader'а качество низкое.
- Составные вопросы («кто и когда») → часто только часть ответа.

## Альтернативная модель (stretch)

`deepset/roberta-base-squad2` — сильнее, медленнее. Скачивается через
`scripts/download_models.py --extras`.
