"""Единая конфигурация проекта: все параметры в одном месте, frozen dataclasses.

Использование:
    from shkerebert.config import default_config, Config
    cfg = default_config()
    # или с переопределением:
    cfg = with_overrides(cfg, chunk__size=128, retriever__top_k=10)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

# Каталог CSV/PNG отчётов eval-скриптов (eval/eval_*.py).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"


@dataclass(frozen=True)
class ChunkConfig:
    """Параметры нарезки документов на чанки."""
    size: int = 256          # целевой размер чанка в токенах эмбеддера
    overlap: int = 64        # перекрытие соседних чанков (токены)
    strategy: str = "sentence"  # "sentence" | "fixed"


@dataclass(frozen=True)
class RetrieverConfig:
    """Параметры поиска (dense + опционально reranker/BM25)."""
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    min_score: float = 0.15       # retrieval-level абстенция: если best < min_score -> "нет ответа"
    normalize: bool = True        # L2-нормировать эмбеддинги (cosine = inner product)

    # Reranker (cross-encoder) — stretch, опционально
    use_reranker: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_pool: int = 20         # сколько кандидатов отдавать в reranker

    # Hybrid BM25 (RRF) — stretch, опционально
    use_bm25: bool = False
    bm25_pool: int = 50           # сколько кандидатов брать из BM25 для фьюза
    rrf_k: int = 60               # константа RRF: 1/(k + rank + 1)


@dataclass(frozen=True)
class ReaderConfig:
    """Параметры экстрактивного ридера (BERT-span)."""
    model: str = "deepset/tinyroberta-squad2"
    max_seq_len: int = 384
    max_answer_len: int = 30
    no_answer_threshold: float = -6.3   # τ: gap = null - best_span; gap > τ => "нет ответа"


@dataclass(frozen=True)
class GeneratorConfig:
    """Параметры генеративного ридера (Qwen GGUF через llama-cpp)."""
    model_path: str = "models/qwen2.5-3b-instruct-q4_k_m.gguf"
    n_ctx: int = 4096
    n_threads: int = 0          # 0 = все доступные
    temperature: float = 0.0
    max_tokens: int = 256
    no_answer_marker: str = "НЕТ ОТВЕТА"


@dataclass(frozen=True)
class Config:
    """Корневая конфигурация пайплайна."""
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    reader: ReaderConfig = field(default_factory=ReaderConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    seed: int = 42
    reader_mode: str = "extractive"   # "extractive" | "generative" | "hybrid"

    def cache_dir(self, subdir: str) -> Path:
        """Директория кэша для данного подтипа (embeddings, index, ...)."""
        root = Path(os.getenv("SHKEREBERT_DATA", Path.home() / ".cache" / "shkerebert"))
        path = root / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path


def default_config() -> Config:
    """Базовая конфигурация (EN, CPU, быстрые модели)."""
    return Config()


def multilingual_config() -> Config:
    """Конфигурация с многоязычным эмбеддером (EN+RU поиск)."""
    return replace(
        default_config(),
        retriever=replace(
            default_config().retriever,
            embed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
    )


def with_overrides(cfg: Config, **kwargs) -> Config:
    """Переопределить вложенные поля через двойное подчёркивание:
       with_overrides(cfg, chunk__size=128, retriever__top_k=10)
    """
    updates: dict[str, Any] = {}
    for key, value in kwargs.items():
        if "__" not in key:
            raise ValueError(f"Ключ должен содержать '__': {key}")
        top, sub = key.split("__", 1)
        if top not in {"chunk", "retriever", "reader", "generator"}:
            raise ValueError(f"Неизвестная секция конфига: {top}")
        section = getattr(cfg, top)
        if not hasattr(section, sub):
            raise ValueError(f"В {top} нет поля {sub}")
        # Собираем замены по секциям
        updates.setdefault(top, {})[sub] = value

    new_cfg = cfg
    for top, subdict in updates.items():
        section = getattr(new_cfg, top)
        new_cfg = replace(new_cfg, **{top: replace(section, **subdict)})
    return new_cfg
