"""Генеративный ридер на локальной Qwen 2.5 Instruct (GGUF, llama-cpp).

Grounded generation: модель получает вопрос + пронумерованные найденные фрагменты и обязана
  * отвечать ТОЛЬКО по контексту,
  * ссылаться на номера фрагментов [1], [2], ...,
  * вернуть маркер отказа, если ответа в контексте нет (обработка «нет ответа»).

Всё локально и офлайн — данные не покидают машину (важно для банковского on-premise).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from functools import lru_cache

from .config import GeneratorConfig
from .chunking import Chunk


@dataclass
class GenAnswer:
    answer: str                       # "" => «нет ответа»
    is_answerable: bool
    cited_indices: list[int] = field(default_factory=list)   # 1-based номера фрагментов
    cited_chunk_ids: list[str] = field(default_factory=list)
    raw: str = ""
    latency_s: float = 0.0


@lru_cache(maxsize=1)
def _load_llm(model_path: str, n_ctx: int, n_threads: int):
    from llama_cpp import Llama

    return Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)


_SYSTEM = (
    "Ты — ассистент, который отвечает на вопрос СТРОГО по предоставленным фрагментам "
    "документа. Правила:\n"
    "1. Используй только информацию из фрагментов ниже. Ничего не выдумывай.\n"
    "2. Сначала напиши сам ответ по существу, а В КОНЦЕ укажи номера использованных "
    "фрагментов в квадратных скобках. Пример: «Столица — Париж [3].»\n"
    "3. НИКОГДА не отвечай одними скобками без текста ответа.\n"
    "4. Если ответа в фрагментах НЕТ — ответь ровно: {marker}\n"
    "5. Отвечай кратко и на языке вопроса."
)


class Generator:
    def __init__(self, cfg: GeneratorConfig):
        self.cfg = cfg
        self.llm = _load_llm(cfg.model_path, cfg.n_ctx, cfg.n_threads)

    def _build_prompt(self, question: str, chunks: list[Chunk]) -> str:
        blocks = []
        for i, ch in enumerate(chunks, 1):
            title = f" ({ch.title})" if ch.title else ""
            blocks.append(f"[{i}]{title} {ch.text}")
        context = "\n\n".join(blocks)
        return f"Фрагменты:\n{context}\n\nВопрос: {question}"

    def generate(self, question: str, chunks: list[Chunk]) -> GenAnswer:
        marker = self.cfg.no_answer_marker
        system = _SYSTEM.format(marker=marker)
        user = self._build_prompt(question, chunks)

        t0 = time.time()
        out = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        dt = time.time() - t0
        text = out["choices"][0]["message"]["content"].strip()

        # Абстенция: маркер отказа встречается в ответе.
        norm = text.upper().replace("Ё", "Е")
        if marker.upper().replace("Ё", "Е") in norm:
            return GenAnswer("", False, raw=text, latency_s=dt)

        # Извлекаем цитаты [n] и мапим на id фрагментов.
        idxs = sorted({int(m) for m in re.findall(r"\[(\d+)\]", text)
                       if 1 <= int(m) <= len(chunks)})
        cited_ids = [chunks[i - 1].id for i in idxs]
        return GenAnswer(
            answer=text, is_answerable=True,
            cited_indices=idxs, cited_chunk_ids=cited_ids,
            raw=text, latency_s=dt,
        )
