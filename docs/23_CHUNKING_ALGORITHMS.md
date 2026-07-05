# Алгоритмы чанкинга: детали реализации

## Sentence-aware (основной)

```python
def _chunk_sentence_aware(text: str, counter: TokenCounter, size: int, overlap: int):
    sentences = split_sentences(text)
    units = []
    for s in sentences:
        n = counter.count(s)
        if n <= size:
            units.append((s, n))
        else:
            # Длинное предложение → режем по токенам
            for piece in _split_by_tokens(s, counter, size, overlap):
                units.append((piece, counter.count(piece)))

    chunks, cur, cur_counts, cur_tokens = [], [], [], 0
    for s, n in units:
        if cur and cur_tokens + n > size:
            chunks.append(" ".join(cur))
            # Overlap: оставляем хвост на ~overlap токенов
            back = 0
            j = len(cur)
            while j > 0 and back < overlap:
                back += cur_counts[j - 1]
                j -= 1
            cur, cur_counts = cur[j:], cur_counts[j:]
            cur_tokens = sum(cur_counts)
            # Если даже overlap-хвост + новое предложение не влезают — чистый старт
            if cur_tokens + n > size:
                cur, cur_counts, cur_tokens = [], [], 0
        cur.append(s)
        cur_counts.append(n)
        cur_tokens += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks
```

### Ключевые свойства

1. **Гарантированное завершение** — for-цикл по `units`, каждое предложение обрабатывается ровно один раз (исправлен баг v1 с бесконечным циклом на предложении без пунктуации).
2. **Overlap работает** — последние предложения предыдущего чанка попадают в следующий.
3. **Fallback для длинных предложений** — если одно предложение > `size`, оно дробится по токенам (`_split_by_tokens`).

## Fixed-token (альтернатива)

```python
def _split_by_tokens(text: str, counter: TokenCounter, size: int, overlap: int):
    ids = counter.encode(text)
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(ids), step):
        window = ids[start : start + size]
        piece = counter.decode(window)
        if piece:
            out.append(piece)
        if start + size >= len(ids):
            break
    return out
```

Скользящее окно по токенам. Не учитывает границы предложений — может резать слова/фразы.

## TokenCounter

```python
class TokenCounter:
    def __init__(self, model_name: str):
        self.tok = AutoTokenizer.from_pretrained(model_name)

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True).strip()
```

- Считает **без спец-токенов** ([CLS], [SEP]) — нас интересует длина полезного текста.
- Кэширует токенируется токенизатор через `@lru_cache` в `_get_tokenizer`.

## Split_sentences

```python
_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+")

def split_sentences(text: str) -> list[str]:
    sentences = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        for s in _SENT_SPLIT.split(para):
            s = s.strip()
            if s:
                sentences.append(s)
    return sentences
```

Простое, но достаточное для энциклопедического текста (SQuAD, Википедия). Для сложных
документов (юридические, с аббревиатурами) можно заменить на `nltk` / `spacy`.

## Чанк (Chunk)

```python
@dataclass(frozen=True)
class Chunk:
    id: str          # "{doc_id}::{index}"
    doc_id: str      # id родительского документа
    text: str        # текст чанка
    index: int       # порядковый номер в документе
    n_tokens: int    # длина в токенах эмбеддера
    title: str       # title документа
    meta: dict       # метаданные документа
```

`id` уникален глобально, используется в FAISS и для цитирования в ответе.

## Тесты (tests/test_chunking.py)

| Тест | Что проверяет |
|---|---|
| `test_split_sentences_basic` | Базовое разбиение по .!? |
| `test_split_sentences_handles_newlines` | Параграфы через \n\n |
| `test_chunk_respects_size_budget` | Чанки не превышают size + запас |
| `test_chunks_have_overlap` | Overlap реально работает (хвост в следующем) |
| `test_chunk_ids_and_indices_are_sequential` | id/index корректны |
| `test_fixed_strategy_covers_all_tokens` | Fixed не теряет токены |
| `test_short_document_single_chunk` | Короткий док = 1 чанк |
| `test_long_sentence_is_split` | Длинное предложение дробится, не зацикливается |
