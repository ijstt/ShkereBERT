"""Разбиение документов на фрагменты (chunks).

Почему это важно: reader видит ограниченное окно, а ретривер работает точнее на
коротких смысловых фрагментах. Слишком крупные чанки => шум и потеря точности поиска;
слишком мелкие => ответ может разрезаться на границе. Поэтому:
  * размер и overlap меряем В ТОКЕНАХ эмбеддера (а не символах) — это то, что реально
    видит модель;
  * overlap не даёт «разрезать» ответ на стыке чанков;
  * есть две стратегии — по предложениям (sentence, дефолт) и по токенам (fixed) —
    чтобы сравнить их в ablation'е.

Обоснование конкретного размера — эмпирическое (см. eval/ablations.py: F1 vs размер).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from .config import ChunkConfig
from .ingest import Document

# Разбиение на предложения: граница после .!? (возможно в кавычках/скобках) + пробел.
# Для чистой прозы SQuAD этого достаточно; сложные случаи не критичны для чанкинга.
_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


@dataclass(frozen=True)
class Chunk:
    id: str          # "{doc_id}::{index}"
    doc_id: str
    text: str
    index: int
    n_tokens: int
    title: str = ""
    meta: dict = field(default_factory=dict)


@lru_cache(maxsize=4)
def _get_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


class TokenCounter:
    """Обёртка над HF-токенизатором для подсчёта/резки по токенам.

    Считаем БЕЗ спец-токенов ([CLS]/[SEP]) — нас интересует длина полезного текста.
    """

    def __init__(self, model_name: str):
        self.tok = _get_tokenizer(model_name)

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True).strip()


def split_sentences(text: str) -> list[str]:
    """Разбить текст на предложения, попутно уважая переводы строк."""
    sentences: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        for s in _SENT_SPLIT.split(para):
            s = s.strip()
            if s:
                sentences.append(s)
    return sentences


def _chunk_sentence_aware(text: str, counter: TokenCounter, size: int, overlap: int):
    """Жадно упаковываем предложения в чанки ~size токенов с overlap.

    Предложение длиннее size режется по токенам (fallback), чтобы не переполнять reader.
    """
    sentences = split_sentences(text)
    # Предсчёт длин; длинные предложения дробим сразу.
    units: list[tuple[str, int]] = []
    for s in sentences:
        n = counter.count(s)
        if n <= size:
            units.append((s, n))
        else:
            for piece in _split_by_tokens(s, counter, size, overlap):
                units.append((piece, counter.count(piece)))

    chunks: list[str] = []
    cur: list[str] = []          # предложения текущего чанка
    cur_counts: list[int] = []   # их длины (чтобы не пересчитывать при overlap)
    cur_tokens = 0
    # For-цикл гарантирует завершение: каждое предложение добавляется ровно один раз.
    for s, n in units:
        if cur and cur_tokens + n > size:
            chunks.append(" ".join(cur))
            # overlap: оставляем хвост предложений на ~overlap токенов.
            back = 0
            j = len(cur)
            while j > 0 and back < overlap:
                back += cur_counts[j - 1]
                j -= 1
            cur, cur_counts = cur[j:], cur_counts[j:]
            cur_tokens = sum(cur_counts)
            # Если даже overlap-хвост + новое предложение не влезают — начинаем чанк с нуля
            # (иначе прогресс не гарантирован). Каждое предложение <= size (длинные уже разбиты).
            if cur_tokens + n > size:
                cur, cur_counts, cur_tokens = [], [], 0
        cur.append(s)
        cur_counts.append(n)
        cur_tokens += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _split_by_tokens(text: str, counter: TokenCounter, size: int, overlap: int):
    """Скользящее окно по токенам: шаг = size - overlap."""
    ids = counter.encode(text)
    if not ids:
        return []
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


def chunk_document(doc: Document, cfg: ChunkConfig, embed_model: str) -> list[Chunk]:
    counter = TokenCounter(embed_model)
    if cfg.strategy == "fixed":
        texts = _split_by_tokens(doc.text, counter, cfg.size, cfg.overlap)
    else:
        texts = _chunk_sentence_aware(doc.text, counter, cfg.size, cfg.overlap)

    chunks = []
    for idx, t in enumerate(texts):
        chunks.append(
            Chunk(
                id=f"{doc.id}::{idx}",
                doc_id=doc.id,
                text=t,
                index=idx,
                n_tokens=counter.count(t),
                title=doc.title,
                meta=dict(doc.meta),
            )
        )
    return chunks


def chunk_documents(docs, cfg: ChunkConfig, embed_model: str) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, cfg, embed_model))
    return out
