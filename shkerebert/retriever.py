"""Поиск релевантных фрагментов для вопроса.

Базовый режим — плотный (dense) поиск top-k по FAISS. Опционально (stretch) кандидаты
переоцениваются кросс-энкодером: он медленнее, но точнее ранжирует, т.к. видит пару
(вопрос, фрагмент) целиком, а не два независимых эмбеддинга.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .chunking import Chunk, chunk_documents
from .config import Config
from .embeddings import Embedder, embed_corpus
from .index import SearchHit, VectorIndex


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
        # Если включён reranker — достаём больший пул кандидатов, затем сужаем до k.
        pool = rcfg.rerank_pool if rcfg.use_reranker else k
        qvec = self.embedder.encode([question])
        hits: list[SearchHit] = self.index.search(qvec, k=pool)[0]

        if not rcfg.use_reranker:
            return [RetrievedChunk(h.chunk, h.score, h.score) for h in hits]

        reranker = _load_reranker(rcfg.rerank_model)
        pairs = [(question, h.chunk.text) for h in hits]
        rerank_scores = reranker.predict(pairs)
        rescored = [
            RetrievedChunk(h.chunk, float(rs), h.score)
            for h, rs in zip(hits, rerank_scores)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:k]
