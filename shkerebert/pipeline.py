"""RAG-пайплайн: вопрос -> ответ + цитата источника, с обработкой «нет ответа».

Поддерживает три режима ридера (выбирает пользователь через cfg.reader_mode или аргумент):
  * "extractive"  — BERT выделяет span (быстро, даёт EM/F1-якорь, порог tau);
  * "generative"  — Qwen генерирует естественный ответ по фрагментам с цитатами;
  * "hybrid"      — Qwen отвечает, а extractive-BERT проверяет обоснованность (groundedness),
                    помечая возможные галлюцинации.

Абстенция «нет ответа» работает во всех режимах:
  1) retrieval-level: лучший фрагмент слишком непохож на вопрос (dense_score < min_score);
  2) reader-level: extractive — по порогу tau; generative — по маркеру отказа модели.

Возвращаем ответ и ССЫЛКИ на использованные фрагменты (требование задания).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config, default_config
from .reader import Reader, SpanAnswer
from .retriever import DenseRetriever, RetrievedChunk


@dataclass
class Source:
    chunk_id: str
    title: str
    text: str
    retrieval_score: float
    is_answer_source: bool = False


@dataclass
class Answer:
    question: str
    answer: str                 # пустая строка => «нет ответа»
    is_answerable: bool
    confidence: float           # extractive: span-null; generative: NaN
    reason: str                 # почему так решили (прозрачность/отладка)
    mode: str = "extractive"
    sources: list[Source] = field(default_factory=list)

    @property
    def display(self) -> str:
        if not self.is_answerable:
            return "В документе нет ответа на этот вопрос."
        return self.answer


class RAGPipeline:
    def __init__(self, retriever: DenseRetriever, cfg: Config,
                 reader: Reader | None = None, generator=None):
        self.retriever = retriever
        self.cfg = cfg
        self._reader = reader          # extractive; грузится лениво
        self._generator = generator    # Qwen; грузится лениво (тяжёлый)

    @classmethod
    def build(cls, docs, cfg: Config | None = None, show_progress: bool = True):
        cfg = cfg or default_config()
        retriever = DenseRetriever.build(docs, cfg, show_progress=show_progress)
        return cls(retriever, cfg)

    # --- ленивые ридеры (не грузим Qwen, пока он не нужен) ---
    @property
    def reader(self) -> Reader:
        if self._reader is None:
            self._reader = Reader(self.cfg.reader)
        return self._reader

    @property
    def generator(self):
        if self._generator is None:
            from .generator import Generator
            self._generator = Generator(self.cfg.generator)
        return self._generator

    def answer(self, question: str, k: int | None = None, mode: str | None = None) -> Answer:
        k = k or self.cfg.retriever.top_k
        mode = mode or self.cfg.reader_mode
        retrieved: list[RetrievedChunk] = self.retriever.retrieve(question, k=k)

        # --- Уровень 1: retrieval-level абстенция (общая для всех режимов) ---
        if not retrieved or retrieved[0].dense_score < self.cfg.retriever.min_score:
            return Answer(question, "", False, float("-inf"),
                          "retrieval below min_score", mode,
                          self._sources(retrieved, None))

        if mode == "generative":
            return self._answer_generative(question, retrieved)
        if mode == "hybrid":
            return self._answer_hybrid(question, retrieved)
        return self._answer_extractive(question, retrieved)

    # --- Режимы ---
    def _answer_extractive(self, question, retrieved) -> Answer:
        spans: list[SpanAnswer] = self.reader.read(question, [r.chunk for r in retrieved])
        best = spans[0]
        confidence = best.score - best.null_score
        tau = self.cfg.reader.no_answer_threshold
        if best.gap > tau or not best.text:
            return Answer(question, "", False, confidence,
                          f"reader no-answer (gap={best.gap:.2f} > tau={tau:.2f})",
                          "extractive", self._sources(retrieved, None))
        return Answer(question, best.text, True, confidence, "answer extracted",
                      "extractive", self._sources(retrieved, best.chunk.id))

    def _answer_generative(self, question, retrieved) -> Answer:
        gen = self.generator.generate(question, [r.chunk for r in retrieved])
        if not gen.is_answerable:
            return Answer(question, "", False, float("nan"),
                          "generator no-answer", "generative",
                          self._sources(retrieved, None))
        return Answer(question, gen.answer, True, float("nan"),
                      f"generated (cites={gen.cited_indices})", "generative",
                      self._sources_multi(retrieved, gen.cited_chunk_ids))

    def _answer_hybrid(self, question, retrieved) -> Answer:
        """Qwen отвечает; extractive-BERT проверяет обоснованность (анти-галлюцинация)."""
        gen = self.generator.generate(question, [r.chunk for r in retrieved])
        spans = self.reader.read(question, [r.chunk for r in retrieved])
        best = spans[0]
        tau = self.cfg.reader.no_answer_threshold
        extractive_supports = (best.gap <= tau) and bool(best.text)

        if not gen.is_answerable:
            # Если генератор отказался, но extractive уверенно нашёл — отдаём extractive.
            if extractive_supports:
                return Answer(question, best.text, True, best.score - best.null_score,
                              "generator abstained; extractive fallback", "hybrid",
                              self._sources(retrieved, best.chunk.id))
            return Answer(question, "", False, float("nan"),
                          "both abstained", "hybrid", self._sources(retrieved, None))

        verdict = "verified by extractive" if extractive_supports else "UNVERIFIED (возможна галлюцинация)"
        return Answer(question, gen.answer, True, float("nan"),
                      f"generated, {verdict}", "hybrid",
                      self._sources_multi(retrieved, gen.cited_chunk_ids))

    # --- Источники ---
    def _sources(self, retrieved, answer_chunk_id):
        return [Source(r.chunk.id, r.chunk.title, r.chunk.text, r.dense_score,
                       r.chunk.id == answer_chunk_id) for r in retrieved]

    def _sources_multi(self, retrieved, answer_chunk_ids):
        ids = set(answer_chunk_ids or [])
        return [Source(r.chunk.id, r.chunk.title, r.chunk.text, r.dense_score,
                       r.chunk.id in ids) for r in retrieved]
