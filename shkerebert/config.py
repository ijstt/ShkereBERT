"""Единая конфигурация ShkereBERT.

Все «магические» параметры (модели, размеры чанков, top-k, порог абстенции, пути)
собраны здесь, чтобы ablation-скрипты могли переопределять их программно, а CLI/UI —
через переменные окружения или аргументы. Значения по умолчанию подобраны под CPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

# --- Пути проекта ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SHKEREBERT_DATA", ROOT / "data"))
RESULTS_DIR = ROOT / "eval" / "results"


@dataclass(frozen=True)
class ChunkConfig:
    """Параметры разбиения документа на фрагменты.

    Размер и overlap меряются в токенах эмбеддера. Значения по умолчанию —
    компромисс «покрытие vs шум», обосновывается ablation'ом по размеру чанка.
    """

    size: int = 256          # целевой размер чанка в токенах
    overlap: int = 64        # перекрытие соседних чанков (чтобы не резать ответ)
    strategy: str = "sentence"  # "sentence" (по предложениям) | "fixed" (по токенам)


@dataclass(frozen=True)
class RetrieverConfig:
    """Параметры плотного поиска (и опционального reranker'а)."""

    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5           # сколько чанков отдаём reader'у
    normalize: bool = True   # L2-нормировка => inner product = cosine
    # Пол похожести top-1: ниже => сразу абстенция (retrieval-level no-answer).
    min_score: float = 0.0
    # Stretch: кросс-энкодер для переоценки кандидатов.
    use_reranker: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_pool: int = 20    # сколько кандидатов достаём до reranker'а


@dataclass(frozen=True)
class ReaderConfig:
    """Параметры экстрактивного QA-reader'а (обучен на SQuAD v2)."""

    model: str = "deepset/tinyroberta-squad2"
    max_seq_len: int = 384
    doc_stride: int = 128
    max_answer_len: int = 64
    top_k_answers: int = 1
    # Порог абстенции reader-level: абстейним, если (null_score - best_span) > tau.
    # Значение откалибровано на dev-сплите SQuAD v2 (eval_e2e): tau* ≈ -6.5 максимизирует
    # общий F1 (EM=76.0, F1=78.3, NoAns_F1=87.7). См. eval/results/e2e_threshold_curve.png.
    no_answer_threshold: float = -6.5


@dataclass(frozen=True)
class Config:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    reader: ReaderConfig = field(default_factory=ReaderConfig)
    data_dir: Path = DATA_DIR
    seed: int = 42

    def cache_dir(self, name: str) -> Path:
        """Каталог кэша для артефакта (эмбеддинги/индекс/корпус)."""
        d = self.data_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d


def default_config() -> Config:
    return Config()


# Утилита для ablation'ов: неглубокое переопределение полей.
def with_overrides(cfg: Config, **kwargs) -> Config:
    """Вернуть копию cfg с изменёнными вложенными полями.

    Пример: with_overrides(cfg, chunk=replace(cfg.chunk, size=128), top_k=3)
    """
    return replace(cfg, **kwargs)
