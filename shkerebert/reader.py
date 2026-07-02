"""Экстрактивный reader: выделяет span-ответ из фрагментов + оценка «нет ответа».

Модель обучена на SQuAD v2 (`deepset/tinyroberta-squad2` по умолчанию) и умеет отвечать
«нет ответа». Мы считаем логиты напрямую, чтобы получить ДВА сопоставимых числа на каждый
фрагмент:
  * best_span_score = max_{i<=j} (start_logit[i] + end_logit[j]) по валидным спанам контекста;
  * null_score      = start_logit[CLS] + end_logit[CLS]  — «скор пустого ответа».

Решение об абстенции принимается по разнице: чем больше (null_score - best_span_score),
тем увереннее «ответа нет». Порог tau калибруется на dev-сплите (см. eval_e2e.py).
Это стандартная схема инференса SQuAD v2, дающая честные EM/F1.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .chunking import Chunk
from .config import ReaderConfig


@dataclass
class SpanAnswer:
    text: str
    score: float          # best_span_score (логит лучшего непустого спана)
    null_score: float     # логит пустого ответа (CLS)
    start_char: int
    end_char: int
    chunk: Chunk

    @property
    def gap(self) -> float:
        """>0 => модель склоняется к «нет ответа» для этого фрагмента."""
        return self.null_score - self.score


@lru_cache(maxsize=2)
def _load_qa(model_name: str):
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.eval()
    return tok, model


class Reader:
    def __init__(self, cfg: ReaderConfig):
        self.cfg = cfg
        self.tokenizer, self.model = _load_qa(cfg.model)

    def _extract(self, start_logits, end_logits, offsets, seq_ids, chunk: Chunk) -> SpanAnswer:
        """Достать лучший span и null-score из логитов ОДНОГО примера (numpy)."""
        cfg = self.cfg
        null_score = float(start_logits[0] + end_logits[0])

        # Валидные токены контекста (sequence_id == 1); паддинг/вопрос исключены.
        ctx_positions = np.where(np.array([sid == 1 for sid in seq_ids]))[0]
        if ctx_positions.size == 0:
            return SpanAnswer("", -1e9, null_score, 0, 0, chunk)

        # Топ-кандидаты по start/end среди контекстных токенов (экономим перебор).
        n_best = 20
        starts = ctx_positions[np.argsort(start_logits[ctx_positions])[-n_best:]]
        ends = ctx_positions[np.argsort(end_logits[ctx_positions])[-n_best:]]

        best = None
        for s in starts:
            for e in ends:
                if e < s or (e - s + 1) > cfg.max_answer_len:
                    continue
                score = float(start_logits[s] + end_logits[e])
                if best is None or score > best[0]:
                    best = (score, int(s), int(e))

        if best is None:
            return SpanAnswer("", -1e9, null_score, 0, 0, chunk)

        score, s, e = best
        start_char, end_char = int(offsets[s][0]), int(offsets[e][1])
        text = chunk.text[start_char:end_char].strip()
        return SpanAnswer(text, score, null_score, start_char, end_char, chunk)

    def _read_one(self, question: str, chunk: Chunk) -> SpanAnswer:
        import torch

        cfg = self.cfg
        enc = self.tokenizer(
            question, chunk.text,
            truncation="only_second", max_length=cfg.max_seq_len,
            return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].numpy()
        seq_ids = enc.sequence_ids(0)
        with torch.no_grad():
            out = self.model(**enc)
        return self._extract(out.start_logits[0].numpy(), out.end_logits[0].numpy(),
                             offsets, seq_ids, chunk)

    def read(self, question: str, chunks: list[Chunk]) -> list[SpanAnswer]:
        """Прочитать ответ из каждого фрагмента, отсортировать по best_span_score.

        Обрабатываем фрагменты по одному (без паддинга): на CPU это быстрее батча —
        паддинг до самого длинного фрагмента добавил бы «пустой» compute (замерено).
        Общая логика извлечения span вынесена в `_extract` (переиспользуется).
        """
        answers = [self._read_one(question, ch) for ch in chunks]
        answers.sort(key=lambda a: a.score, reverse=True)
        return answers
