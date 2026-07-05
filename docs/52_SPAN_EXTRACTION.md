# Извлечение span-ответа: детали алгоритма

## Проблема

BERT выдаёт логиты начала и конца для **каждого токена** последовательности:
- `start_logits[seq_len]`
- `end_logits[seq_len]`

Нужно найти **лучший валидный span** (start ≤ end, длина ≤ max_answer_len)
только среди токенов **контекста** (не вопроса, не паддинга, не [CLS]/[SEP]).

## Алгоритм (_extract в reader.py)

```python
def _extract(self, start_logits, end_logits, offsets, seq_ids, chunk: Chunk):
    # 1. Null-score = логит [CLS] (токен 0)
    null_score = float(start_logits[0] + end_logits[0])

    # 2. Индексы токенов контекста (sequence_id == 1)
    ctx_positions = np.where(np.array([sid == 1 for sid in seq_ids]))[0]
    if ctx_positions.size == 0:
        return SpanAnswer("", -1e9, null_score, 0, 0, chunk)

    # 3. Топ-кандидаты по start/end отдельно (экономия O(n²) → O(k²))
    n_best = 20
    starts = ctx_positions[np.argsort(start_logits[ctx_positions])[-n_best:]]
    ends   = ctx_positions[np.argsort(end_logits[ctx_positions])[-n_best:]]

    # 4. Перебор пар (start, end) с валидацией
    best = None
    for s in starts:
        for e in ends:
            if e < s:                          # конец до начала
                continue
            if (e - s + 1) > self.cfg.max_answer_len:  # слишком длинный
                continue
            score = float(start_logits[s] + end_logits[e])
            if best is None or score > best[0]:
                best = (score, int(s), int(e))

    if best is None:
        return SpanAnswer("", -1e9, null_score, 0, 0, chunk)

    # 5. Маппинг токенов → символы в исходном тексте чанка
    score, s, e = best
    start_char, end_char = int(offsets[s][0]), int(offsets[e][1])
    text = chunk.text[start_char:end_char].strip()
    return SpanAnswer(text, score, null_score, start_char, end_char, chunk)
```

## Ключевые решения

| Решение | Почему |
|---|---|
| **Топ-20 по start/end отдельно** | Полный перебор O(n²) на 384 токенах = 147k пар; топ-20 даёт 400 пар — ×360 ускорение без потери качества (лучший span почти всегда в топе по обоим логитам) |
| **sequence_id == 1** | `truncation="only_second"` делает: вопрос=0, контекст=1, паддинг=None. Исключаем вопрос и спец-токены |
| **max_answer_len=30** | SQuAD ответы короткие (медиана 2–3 токена); длинные спаны — шум |
| **offsets_mapping** | Токенизатор возвращает `(start_char, end_char)` для каждого токена → точное извлечение подстроки из `chunk.text` |
| **strip()** | Убираем пробелы на границах, которые мог добавить токенизатор |

## Gap = null_score - best_span_score

```python
@property
def gap(self) -> float:
    return self.null_score - self.score
```

- `gap > 0` → модель считает «пустой ответ» более вероятным
- `gap < 0` → модель нашла достойный span
- Порог τ калибруется: `gap > τ` → абстенция

## Почему НЕ батчинг на CPU

Попытка батчить `k` чанков в один forward:
- Паддинг до длины самого длинного чанка → много пустых токенов
- На CPU матричные умножения на нули всё равно стоят циклов
- **Замерено**: пофрагментная обработка быстрее батча на 10–20% (см. CHANGELOG v2.1)

## Тестирование

Юнит-тесты в `tests/test_retriever.py` (проверяют интеграцию retriever+reader),
специфичные для span — в интеграционных прогонах `eval/eval_e2e.py`.
