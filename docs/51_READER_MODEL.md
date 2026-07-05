# Экстрактивный ридер: выбор модели

## Выбор: deepset/tinyroberta-squad2

| Модель | Параметры | F1 (SQuAD v2 dev) | Скорость (CPU) | Язык |
|---|---|---|---|---|
| **tinyroberta-squad2** | **14M** | **78.3** | **~50 мс/чанк** | **EN** |
| roberta-base-squad2 | 125M | 83.5 | ~200 мс/чанк | EN |
| deberta-v3-large-squad2 | 300M | 89.0 | ~800 мс/чанк | EN |
| xlm-roberta-base-squad2 | 270M | 71.2 (EN) / 68 (RU) | ~300 мс/чанк | Multi |

### Почему tinyroberta

1. **Скорость на CPU** — 14M параметров = ~50 мс forward pass на чанк. При top-k=5
   полный ридер ~250 мс. Крупные модели дают ×4–16 замедление.
2. **Достаточное качество** — 78.3 F1 на dev SQuAD v2 (близко к roberta-base 83.5
   при 1/9 параметров).
3. **SQuAD v2 нативная** — обучена с unanswerable вопросами, выдаёт null-score.
4. **Стабильность** — нет проблем с загрузкой/токенизацией (в отличие от некоторых
   multilingual чекпоинтов).

### Почему НЕ ruBERT / xlm-roberta для baseline

- **На EN теряют 7–8 пунктов F1** — SQuAD v2 метрика считается на EN.
- **Multilingual reader** на EN даёт ~71 F1 против 78 у tinyroberta.
- Для RU используется **generative режим (Qwen)** — он двуязычный и качественнее
  на свободной генерации, чем экстрактивный multilingual BERT.

## Архитектура ридера (reader.py)

```python
class Reader:
    def __init__(self, cfg: ReaderConfig):
        self.tokenizer, self.model = _load_qa(cfg.model)  # cached

    def _read_one(self, question: str, chunk: Chunk) -> SpanAnswer:
        # Токенизация: question + chunk.text (truncation="only_second")
        # Forward → start_logits, end_logits
        # Извлечение лучшего span + null_score
        return self._extract(start_logits, end_logits, offsets, seq_ids, chunk)

    def read(self, question: str, chunks: list[Chunk]) -> list[SpanAnswer]:
        # Пофрагментная обработка (НЕ батч!) — на CPU быстрее без паддинга
        answers = [self._read_one(question, ch) for ch in chunks]
        answers.sort(key=lambda a: a.score, reverse=True)
        return answers
```

## SpanAnswer

```python
@dataclass
class SpanAnswer:
    text: str           # текст ответа (пусто = нет ответа)
    score: float        # best_span_score (start_logit + end_logit)
    null_score: float   # логит [CLS] = "оценка пустого ответа"
    start_char: int     # смещение в chunk.text
    end_char: int
    chunk: Chunk

    @property
    def gap(self) -> float:
        return self.null_score - self.score  # >0 => склонность к "нет ответа"
```

## Null-score и калибровка τ

SQuAD v2 модель обучена предсказывать «пустой ответ» через токен [CLS].
- `null_score = start_logit[0] + end_logit[0]`
- `best_span_score = max_{i≤j} (start_logit[i] + end_logit[j])` по валидным спанам
- `gap = null_score - best_span_score`

Если `gap > τ` → «нет ответа». τ калибруется на calibration-сплите (см. `93_E2E_EVAL.md`).

## Doc-stride (длинные контексты)

Если чанки)

```python
# В _read_one: truncation="only_second", max_length=384
# Если chunk.text > 384 токенов — tokenizer делает sliding window
# с stride (по умолчанию 128) и агрегирует логиты по max
```

Позволяет читать чанки длиннее max_seq_len. В нашем пайплайне чанки ≤ 256 токенов,
поэтому doc-stride не срабатывает (но код готов).

## Загрузка модели (кэш)

```python
@lru_cache(maxsize=2)
def _load_qa(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.eval()
    return tok, model
```

Кэш на уровне процесса — при многократных вызовах `Reader()` модель не перезагружается.
