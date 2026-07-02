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

    # Эмбеддер можно переключить через env SHKEREBERT_EMBED (напр. на многоязычный для RU).
    embed_model: str = os.environ.get(
        "SHKEREBERT_EMBED", "sentence-transformers/all-MiniLM-L6-v2")
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
    # Откалибровано на CALIBRATION-сплите SQuAD v2 (eval_e2e, без утечки в test): tau* ≈ -6.3.
    # На held-out test: EM=73.6, F1=75.8, NoAns_F1=84.6 (oracle-контекст: F1=81.0).
    no_answer_threshold: float = -6.3


@dataclass(frozen=True)
class GeneratorConfig:
    """Генеративный ридер (Qwen 2.5 Instruct, GGUF, локально через llama-cpp).

    Полностью офлайн: ни один документ/вопрос не покидает машину — ключевой аргумент
    для банковского on-premise-сценария. Путь к модели переопределяется env SHKEREBERT_LLM.
    """

    # По умолчанию ищем модель в ./models проекта; переопределяется env SHKEREBERT_LLM.
    model_path: str = os.environ.get(
        "SHKEREBERT_LLM",
        str(ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"),
    )
    n_ctx: int = 4096
    n_threads: int = 8
    max_tokens: int = 256
    temperature: float = 0.0     # детерминированность для воспроизводимости/оценки
    # Маркер отказа, который модель обязана вернуть, если ответа в контексте нет.
    no_answer_marker: str = "НЕТ ОТВЕТА"


@dataclass(frozen=True)
class Config:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    reader: ReaderConfig = field(default_factory=ReaderConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    # Режим ридера: "extractive" (BERT) | "generative" (Qwen) | "hybrid" (Qwen + BERT-проверка).
    reader_mode: str = "extractive"
    data_dir: Path = DATA_DIR
    seed: int = 42

    def cache_dir(self, name: str) -> Path:
        """Каталог кэша для артефакта (эмбеддинги/индекс/корпус)."""
        d = self.data_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d


def default_config() -> Config:
    return Config()


def multilingual_config() -> Config:
    """Пресет EN+RU (напр. банковские документы). Меняет эмбеддер на многоязычный.

    Требует загруженной модели paraphrase-multilingual-MiniLM-L12-v2 (эмбеддинги EN+RU).
    Генеративный ридер (Qwen) сам по себе уже двуязычный.
    """
    return Config(
        retriever=RetrieverConfig(
            embed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            top_k=Config().retriever.top_k,
        )
    )


# Утилита для ablation'ов: неглубокое переопределение полей.
def with_overrides(cfg: Config, **kwargs) -> Config:
    """Вернуть копию cfg с изменёнными вложенными полями.

    Пример: with_overrides(cfg, chunk=replace(cfg.chunk, size=128), top_k=3)
    """
    return replace(cfg, **kwargs)
