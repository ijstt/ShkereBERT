"""Поиск релевантных фрагментов для вопроса.

Базовый режим — плотный (dense) поиск top-k по FAISS. Два stretch-усиления:
  * reranker (кросс-энкодер): точнее ранжирует, т.к. видит пару (вопрос, фрагмент)
    целиком, а не два независимых эмбеддинга; медленнее;
  * hybrid BM25: словарный поиск ловит точные термины/номера, где dense «плывёт»;
    сливаем два ранжирования через Reciprocal Rank Fusion (шкалы скоров несравнимы,
    поэтому фьюзим ранги, а не скоры).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .chunking import Chunk, chunk_documents
from .config import Config
from .embeddings import Embedder, embed_corpus
from .index import SearchHit, VectorIndex

_WORD = re.compile(r"[\w']+", re.UNICODE)


def _bm25_tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float          # финальный скор (dense или rerank)
    dense_score: float    # исходный dense-скор (для диагностики/абстенции)


@lru_cache(maxsize=1)
def _load_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device="cpu")


class DenseRetriever:
    def __init__(self, index: VectorIndex, embedder: Embedder, cfg: Config):
        self.index = index
        self.embedder = embedder
        self.cfg = cfg
        self._bm25 = None  # строится лениво при первом гибридном запросе
        self._pos = None   # chunk.id -> позиция в корпусе (для RRF-слияния)

    @property
    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([_bm25_tokens(c.text) for c in self.index.chunks])
        return self._bm25

    @classmethod
    def build(cls, docs, cfg: Config, show_progress: bool = True) -> "DenseRetriever":
        """Собрать retriever из документов: чанкинг -> эмбеддинги -> индекс."""
        chunks = chunk_documents(docs, cfg.chunk, cfg.retriever.embed_model)
        embeddings, embedder = embed_corpus(chunks, cfg, show_progress=show_progress)
        index = VectorIndex(chunks, embeddings)
        return cls(index, embedder, cfg)

    def retrieve(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
        rcfg = self.cfg.retriever
        k = k or rcfg.top_k
        # Если включены reranker/BM25 — достаём больший пул кандидатов, затем сужаем до k.
        pool = k
        if rcfg.use_reranker:
            pool = max(pool, rcfg.rerank_pool)
        if rcfg.use_bm25:
            pool = max(pool, rcfg.bm25_pool)
        qvec = self.embedder.encode([question])
        hits: list[SearchHit] = self.index.search(qvec, k=pool)[0]
        # dense_score сохраняем отдельно при любом слиянии/переоценке: на нём живёт
        # retrieval-level абстенция (см. pipeline), его шкала — косинус.
        cands = [RetrievedChunk(h.chunk, h.score, h.score) for h in hits]

        if rcfg.use_bm25:
            cands = self._fuse_bm25(question, qvec[0], cands)

        if rcfg.use_reranker:
            reranker = _load_reranker(rcfg.rerank_model)
            scores = reranker.predict([(question, c.chunk.text) for c in cands])
            cands = [RetrievedChunk(c.chunk, float(s), c.dense_score)
                     for c, s in zip(cands, scores)]
            cands.sort(key=lambda r: r.score, reverse=True)

        return cands[:k]

    def _fuse_bm25(self, question: str, qvec: np.ndarray,
                   dense_cands: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Слияние dense-ранжирования с BM25 через Reciprocal Rank Fusion.

        RRF(chunk) = сумма по спискам 1/(rrf_k + rank+1). Фьюзим ранги, а не скоры:
        косинус и BM25 в несравнимых шкалах. Кандидатам, пришедшим только из BM25,
        dense-скор доставаем реконструкцией вектора из FAISS (дёшево, их <= bm25_pool).
        """
        rcfg = self.cfg.retriever
        chunks = self.index.chunks
        if self._pos is None:
            self._pos = {c.id: i for i, c in enumerate(chunks)}

        bm_scores = self.bm25.get_scores(_bm25_tokens(question))
        bm_top = [int(i) for i in np.argsort(bm_scores)[::-1][: rcfg.bm25_pool]
                  if bm_scores[i] > 0]

        rrf: dict[int, float] = {}
        dense: dict[int, float] = {}
        for rank, c in enumerate(dense_cands):
            i = self._pos[c.chunk.id]
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (rcfg.rrf_k + rank + 1)
            dense[i] = c.dense_score
        for rank, i in enumerate(bm_top):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (rcfg.rrf_k + rank + 1)
            if i not in dense:
                dense[i] = float(np.dot(qvec, self.index.index.reconstruct(i)))

        order = sorted(rrf, key=rrf.get, reverse=True)
        return [RetrievedChunk(chunks[i], rrf[i], dense[i]) for i in order]
