"""Векторный индекс (FAISS) над эмбеддингами чанков.

Используем `IndexFlatIP` — точный поиск по inner product. При L2-нормированных векторах
inner product = cosine similarity. Точный (не приближённый) индекс уместен: корпус SQuAD
невелик (десятки тысяч чанков), а точность важнее скорости для учебной оценки.
Индекс и чанки сохраняются на диск, чтобы не пересобирать каждый запуск.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


class VectorIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray):
        import faiss

        assert embeddings.shape[0] == len(chunks), "чанки и эмбеддинги рассинхронизированы"
        self.chunks = chunks
        self.dim = int(embeddings.shape[1])
        self._faiss = faiss
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(np.ascontiguousarray(embeddings.astype("float32")))

    def search(self, query_vecs: np.ndarray, k: int = 5) -> list[list[SearchHit]]:
        if query_vecs.ndim == 1:
            query_vecs = query_vecs[None, :]
        q = np.ascontiguousarray(query_vecs.astype("float32"))
        scores, idx = self.index.search(q, k)
        results: list[list[SearchHit]] = []
        for row_scores, row_idx in zip(scores, idx):
            hits = [
                SearchHit(chunk=self.chunks[i], score=float(s))
                for s, i in zip(row_scores, row_idx)
                if i != -1
            ]
            results.append(hits)
        return results

    # --- Сохранение / загрузка --------------------------------------------
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, directory: str | Path) -> "VectorIndex":
        import faiss

        directory = Path(directory)
        with open(directory / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        obj = cls.__new__(cls)
        obj._faiss = faiss
        obj.chunks = chunks
        obj.index = faiss.read_index(str(directory / "index.faiss"))
        obj.dim = obj.index.d
        return obj
