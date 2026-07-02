"""Векторные представления фрагментов и вопросов + дисковый кэш.

Эмбеддинги считаются один раз и кэшируются на диск: на CPU это самая дорогая операция,
а корпус SQuAD статичен. Ключ кэша учитывает модель, стратегию чанкинга и сигнатуру
корпуса, поэтому при смене параметров кэш не «протухает» незаметно.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import Config
from .chunking import Chunk


@lru_cache(maxsize=2)
def _load_st_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    # device="cpu" — явно, чтобы не пытаться на GPU при CUDA-сборке torch.
    return SentenceTransformer(model_name, device="cpu")


class Embedder:
    def __init__(self, model_name: str, normalize: bool = True):
        self.model_name = model_name
        self.normalize = normalize
        self.model = _load_st_model(model_name)

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts, batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        vecs = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
        )
        return vecs.astype("float32")


def _corpus_signature(chunks: list[Chunk], cfg: Config) -> str:
    """Стабильный хэш корпуса+параметров для ключа кэша."""
    h = hashlib.sha1()
    h.update(cfg.retriever.embed_model.encode())
    h.update(f"{cfg.chunk.size}-{cfg.chunk.overlap}-{cfg.chunk.strategy}".encode())
    h.update(str(len(chunks)).encode())
    # Хэшируем id первых/последних чанков + общий id-поток (дёшево и достаточно устойчиво).
    for ch in chunks:
        h.update(ch.id.encode())
    return h.hexdigest()[:16]


def embed_corpus(chunks: list[Chunk], cfg: Config, show_progress: bool = True):
    """Вернуть (embeddings[N,dim] float32, embedder), используя дисковый кэш."""
    embedder = Embedder(cfg.retriever.embed_model, normalize=cfg.retriever.normalize)
    sig = _corpus_signature(chunks, cfg)
    cache = cfg.cache_dir("embeddings")
    vec_path = cache / f"{sig}.npy"
    meta_path = cache / f"{sig}.json"

    if vec_path.exists() and meta_path.exists():
        vecs = np.load(vec_path)
        if vecs.shape[0] == len(chunks):
            return vecs, embedder

    vecs = embedder.encode([c.text for c in chunks], show_progress=show_progress)
    np.save(vec_path, vecs)
    meta_path.write_text(
        json.dumps(
            {
                "model": cfg.retriever.embed_model,
                "n_chunks": len(chunks),
                "dim": int(vecs.shape[1]),
                "chunk_cfg": vars(cfg.chunk),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return vecs, embedder
