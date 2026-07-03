"""Интеграционный тест плотного поиска: релевантный документ должен ранжироваться выше.

Использует реальный (лёгкий) эмбеддер all-MiniLM-L6-v2; модель кэшируется, тест быстрый.
"""

from __future__ import annotations

from shkerebert.config import default_config
from shkerebert.ingest import Document
from shkerebert.retriever import DenseRetriever


def test_relevant_document_ranked_first():
    docs = [
        Document(id="d-cats", text="Cats are small domesticated carnivorous mammals. "
                                   "They are often kept as pets and are known for purring."),
        Document(id="d-python", text="Python is a high-level programming language known "
                                     "for readable syntax and a large standard library."),
        Document(id="d-eiffel", text="The Eiffel Tower is a wrought-iron lattice tower in "
                                     "Paris, France, built for the 1889 World's Fair."),
    ]
    cfg = default_config()
    retriever = DenseRetriever.build(docs, cfg, show_progress=False)

    hits = retriever.retrieve("Which programming language has readable syntax?", k=3)
    assert hits[0].chunk.doc_id == "d-python"

    hits = retriever.retrieve("Where is the Eiffel Tower located?", k=3)
    assert hits[0].chunk.doc_id == "d-eiffel"


def test_retrieve_respects_k():
    docs = [Document(id=f"d{i}", text=f"Document number {i} about topic {i}." * 5)
            for i in range(6)]
    cfg = default_config()
    retriever = DenseRetriever.build(docs, cfg, show_progress=False)
    hits = retriever.retrieve("topic 3", k=2)
    assert len(hits) == 2
    # dense_score убывает по рангу
    assert hits[0].dense_score >= hits[1].dense_score


def test_bm25_hybrid_finds_exact_term():
    """RRF-гибрид: точный редкий термин должен подтянуть нужный документ наверх,
    а dense_score у кандидатов остаться в шкале косинуса (для абстенции)."""
    from dataclasses import replace

    docs = [
        Document(id="d-tariff", text="Tariff code ZX-9917 applies a 0.75 percent fee "
                                     "for corporate wire transfers above the limit."),
        Document(id="d-cats", text="Cats are small domesticated carnivorous mammals "
                                   "often kept as pets and known for purring sounds."),
        Document(id="d-python", text="Python is a programming language with readable "
                                     "syntax and a large standard library for scripts."),
    ]
    cfg = default_config()
    cfg = replace(cfg, retriever=replace(cfg.retriever, use_bm25=True))
    retriever = DenseRetriever.build(docs, cfg, show_progress=False)

    hits = retriever.retrieve("What fee does tariff ZX-9917 apply?", k=3)
    assert hits[0].chunk.doc_id == "d-tariff"
    # dense_score — косинус, а не RRF: нормированные векторы дают |cos| <= 1
    assert all(-1.0 <= h.dense_score <= 1.0 for h in hits)
    # финальный скор — RRF, убывает по рангу
    assert hits[0].score >= hits[-1].score
