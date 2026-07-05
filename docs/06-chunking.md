# 06. Чанкинг

Модуль `shkerebert/chunking.py`.

## Зачем нужен чанкинг

1. **Reader** (BERT) видит максимум ~384 токена на пару (вопрос + контекст).
2. **Retriever** точнее на коротких смысловых фрагментах.
3. **Компромисс:**
   - слишком крупные чанки → шум, низкий Recall@1;
   - слишком мелкие → ответ может разрезаться; больше чанков → медленнее индекс.

Размер и overlap измеряются **в токенах эмбеддера** — это то, что реально «видит» модель
при индексации и поиске.

## Chunk

```29:37:shkerebert/chunking.py
@dataclass(frozen=True)
class Chunk:
    id: str          # "{doc_id}::{index}"
    doc_id: str
    text: str
    index: int
    n_tokens: int
    title: str = ""
    meta: dict = field(default_factory=dict)
```

## TokenCounter

```47:63:shkerebert/chunking.py
class TokenCounter:
    def __init__(self, model_name: str):
        self.tok = _get_tokenizer(model_name)

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))
```

Токенизатор берётся из той же HF-модели, что и эмбеддер (`cfg.retriever.embed_model`).
Спец-токены `[CLS]`/`[SEP]` **не** считаются — только полезный текст.

Токенизатор кэшируется через `@lru_cache(maxsize=4)` в `_get_tokenizer`.

## Разбиение на предложения

```24:26:shkerebert/chunking.py
_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
```

```66:77:shkerebert/chunking.py
def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for para in text.split("\n"):
        ...
        for s in _SENT_SPLIT.split(para):
```

- Уважает абзацы (`\n\n`).
- Граница после `.!?`, возможно с закрывающими кавычками/скобками.
- Для прозы SQuAD достаточно; edge cases (Mr., U.S.A.) не критичны.

## Стратегия `sentence` (default)

```80:121:shkerebert/chunking.py
def _chunk_sentence_aware(text, counter, size, overlap):
    sentences = split_sentences(text)
    # Длинные предложения (> size) дробятся через _split_by_tokens
    ...
    for s, n in units:
        if cur and cur_tokens + n > size:
            chunks.append(" ".join(cur))
            # overlap: хвост предложений на ~overlap токенов
            ...
        cur.append(s)
```

**Алгоритм:**

1. Разбить текст на предложения.
2. Жадно упаковывать предложения, пока сумма токенов ≤ `size`.
3. При переполнении — сохранить чанк, оставить хвост на `overlap` токенов.
4. Предложение длиннее `size` — fallback на `_split_by_tokens`.

**Гарантия прогресса:** каждое предложение добавляется ровно один раз; длинные уже
разбиты на куски ≤ size.

## Стратегия `fixed`

```124:138:shkerebert/chunking.py
def _split_by_tokens(text, counter, size, overlap):
    ids = counter.encode(text)
    step = max(1, size - overlap)
    for start in range(0, len(ids), step):
        window = ids[start : start + size]
        piece = counter.decode(window)
```

Скользящее окно по токенам, шаг = `size - overlap`. Не уважает границы предложений —
используется для ablation-сравнения.

## Точка входа

```141:168:shkerebert/chunking.py
def chunk_document(doc, cfg, embed_model):
    counter = TokenCounter(embed_model)
    if cfg.strategy == "fixed":
        texts = _split_by_tokens(...)
    else:
        texts = _chunk_sentence_aware(...)
    for idx, t in enumerate(texts):
        chunks.append(Chunk(id=f"{doc.id}::{idx}", ...))
```

## Baseline-параметры и ablation

| chunk_size | Recall@1 | best F1 | Комментарий |
|------------|----------|---------|-------------|
| 128 | 0.854 | 78.1 | Лучший recall, больше чанков |
| **256** | 0.829 | 77.5 | **Baseline** — баланс |
| 512 | 0.824 | 77.3 | Крупные чанки «шумят» |

Overlap baseline: **64 токена** (~25% от 256).

## Особенность SQuAD: короткие контексты

Медиана контекста SQuAD ~141 токен. При `chunk_size=256` режется только ~8% документов.
Вывод «размер чанка почти не влияет» на коротких контекстах **нельзя** переносить на
длинные PDF — для этого есть `eval/eval_longdoc.py` (склейка статей Wikipedia по title).

## Тесты

`tests/test_chunking.py` проверяет:

- разбиение предложений и `\n`;
- бюджет токенов (`n_tokens <= size + 32`);
- наличие overlap между соседними чанками;
- sequential id/index;
- покрытие текста в `fixed`-стратегии;
- дробление одного длинного «предложения».
