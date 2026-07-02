"""RAG-пайплайн: вопрос -> ответ + цитата источника, с обработкой «нет ответа».

Двухуровневая абстенция:
  1) retrieval-level: если лучший фрагмент слишком непохож на вопрос (dense_score ниже
     порога) — сразу «нет ответа»;
  2) reader-level: если null_score - best_span_score > tau — модель считает, что ответа
     в найденных фрагментах нет.

Иначе возвращаем span-ответ и ССЫЛКУ на фрагмент, из которого он извлечён (требование
задания «вернуть ответ и ссылку на использованный фрагмент»).
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
    confidence: float           # best_span_score - null_score (чем больше, тем увереннее)
    reason: str                 # почему так решили (для прозрачности/отладки)
    sources: list[Source] = field(default_factory=list)

    @property
    def display(self) -> str:
        if not self.is_answerable:
            return "В документе нет ответа на этот вопрос."
        return self.answer


class RAGPipeline:
    def __init__(self, retriever: DenseRetriever, reader: Reader, cfg: Config):
        self.retriever = retriever
        self.reader = reader
        self.cfg = cfg

    @classmethod
    def build(cls, docs, cfg: Config | None = None, show_progress: bool = True):
        cfg = cfg or default_config()
        retriever = DenseRetriever.build(docs, cfg, show_progress=show_progress)
        reader = Reader(cfg.reader)
        return cls(retriever, reader, cfg)

    def answer(self, question: str, k: int | None = None) -> Answer:
        k = k or self.cfg.retriever.top_k
        retrieved: list[RetrievedChunk] = self.retriever.retrieve(question, k=k)

        # --- Уровень 1: retrieval-level абстенция ---
        if not retrieved or retrieved[0].dense_score < self.cfg.retriever.min_score:
            return Answer(
                question=question,
                answer="",
                is_answerable=False,
                confidence=float("-inf"),
                reason="retrieval below min_score",
                sources=self._sources(retrieved, answer_chunk_id=None),
            )

        # --- Reader по найденным фрагментам ---
        spans: list[SpanAnswer] = self.reader.read(question, [r.chunk for r in retrieved])
        best = spans[0]
        confidence = best.score - best.null_score  # = -gap

        # --- Уровень 2: reader-level абстенция ---
        tau = self.cfg.reader.no_answer_threshold
        if best.gap > tau or not best.text:
            return Answer(
                question=question,
                answer="",
                is_answerable=False,
                confidence=confidence,
                reason=f"reader no-answer (gap={best.gap:.2f} > tau={tau:.2f})",
                sources=self._sources(retrieved, answer_chunk_id=None),
            )

        return Answer(
            question=question,
            answer=best.text,
            is_answerable=True,
            confidence=confidence,
            reason="answer extracted",
            sources=self._sources(retrieved, answer_chunk_id=best.chunk.id),
        )

    def _sources(self, retrieved: list[RetrievedChunk], answer_chunk_id: str | None):
        out = []
        for r in retrieved:
            out.append(
                Source(
                    chunk_id=r.chunk.id,
                    title=r.chunk.title,
                    text=r.chunk.text,
                    retrieval_score=r.dense_score,
                    is_answer_source=(r.chunk.id == answer_chunk_id),
                )
            )
        return out
