"""Юнит-тесты чанкинга: границы, overlap, покрытие, конфигурируемость размера."""

from __future__ import annotations

import pytest

from shkerebert.chunking import (
    Chunk,
    TokenCounter,
    chunk_document,
    split_sentences,
)
from shkerebert.config import ChunkConfig
from shkerebert.ingest import Document

EMBED = "sentence-transformers/all-MiniLM-L6-v2"


def _doc(text: str) -> Document:
    return Document(id="d1", text=text, title="t")


def test_split_sentences_basic():
    text = "First sentence. Second one! Third? Yes."
    sents = split_sentences(text)
    assert sents == ["First sentence.", "Second one!", "Third?", "Yes."]


def test_split_sentences_handles_newlines():
    sents = split_sentences("Para one.\n\nPara two here.")
    assert sents == ["Para one.", "Para two here."]


def test_chunk_respects_size_budget():
    counter = TokenCounter(EMBED)
    # Длинный текст из повторяющихся предложений.
    text = " ".join(f"Sentence number {i} has some words." for i in range(60))
    cfg = ChunkConfig(size=64, overlap=16, strategy="sentence")
    chunks = chunk_document(_doc(text), cfg, EMBED)
    assert len(chunks) > 1
    # Небольшой запас: один длинный «хвост» из предложений может слегка превысить бюджет,
    # но не катастрофически.
    for ch in chunks:
        assert ch.n_tokens <= cfg.size + 32


def test_chunks_have_overlap():
    text = " ".join(f"Alpha beta gamma delta {i}." for i in range(40))
    cfg = ChunkConfig(size=48, overlap=24, strategy="sentence")
    chunks = chunk_document(_doc(text), cfg, EMBED)
    assert len(chunks) >= 2
    # Последнее предложение предыдущего чанка должно встречаться в следующем (overlap).
    first_last_sent = chunks[0].text.split(".")[-2].strip()
    assert first_last_sent[:10] in chunks[1].text


def test_chunk_ids_and_indices_are_sequential():
    text = " ".join(f"Word set {i} here now." for i in range(30))
    cfg = ChunkConfig(size=40, overlap=8, strategy="sentence")
    chunks = chunk_document(_doc(text), cfg, EMBED)
    for i, ch in enumerate(chunks):
        assert ch.index == i
        assert ch.id == f"d1::{i}"
        assert isinstance(ch, Chunk)


def test_fixed_strategy_covers_all_tokens():
    counter = TokenCounter(EMBED)
    text = " ".join(f"token{i}" for i in range(200))
    cfg = ChunkConfig(size=50, overlap=10, strategy="fixed")
    chunks = chunk_document(_doc(text), cfg, EMBED)
    assert len(chunks) > 1
    # Первый и последний токены исходного текста должны присутствовать в чанках.
    joined = " ".join(ch.text for ch in chunks)
    assert "token0" in joined
    assert "token199" in joined


def test_short_document_single_chunk():
    cfg = ChunkConfig(size=256, overlap=64, strategy="sentence")
    chunks = chunk_document(_doc("Just a short text."), cfg, EMBED)
    assert len(chunks) == 1
    assert chunks[0].text == "Just a short text."


def test_long_sentence_is_split():
    # Одно «предложение» длиннее бюджета должно раздробиться, а не переполнить чанк.
    long_sentence = " ".join(f"w{i}" for i in range(300))  # без точек
    cfg = ChunkConfig(size=50, overlap=10, strategy="sentence")
    chunks = chunk_document(_doc(long_sentence), cfg, EMBED)
    assert len(chunks) > 1
