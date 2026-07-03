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


# System-prompt подбирается под язык вопроса: русский prompt на EN-вопросах провоцировал
# Qwen отвечать по-русски (найдено оценкой eval_generative). Правило «только по фрагментам»
# усилено явным запретом внешних знаний: на unanswerable-вопросах модель охотно отвечала
# из собственной памяти (правдоподобно, но это галлюцинация по отношению к документу).
_SYSTEM_RU = (
    "Ты — ассистент, который отвечает на вопрос СТРОГО по предоставленным фрагментам "
    "документа. Правила:\n"
    "1. Используй только информацию из фрагментов ниже. Никаких внешних знаний, ничего "
    "не выдумывай, даже если знаешь ответ из других источников.\n"
    "2. Сначала напиши сам ответ по существу, а В КОНЦЕ укажи номера использованных "
    "фрагментов в квадратных скобках. Пример: «Столица — Париж [3].»\n"
    "3. НИКОГДА не отвечай одними скобками без текста ответа.\n"
    "4. Если ответа в фрагментах НЕТ — ответь ровно: {marker}\n"
    "5. Отвечай кратко и на языке вопроса."
)

_SYSTEM_EN = (
    "You are an assistant that answers the question STRICTLY from the provided document "
    "fragments. Rules:\n"
    "1. Use only information from the fragments below. No outside knowledge; do not "
    "invent facts even if you know the answer from elsewhere.\n"
    "2. Write the answer itself first, and at the END cite the used fragment numbers "
    "in square brackets. Example: \"The capital is Paris [3].\"\n"
    "3. NEVER reply with brackets alone, without answer text.\n"
    "4. If the fragments do NOT contain the answer, reply exactly: {marker}\n"
    "5. Answer briefly, in the language of the question."
)

_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)
# Маркер отказа для EN-промпта; при детекции абстенции принимаем оба варианта.
_EN_MARKER = "NO ANSWER"


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
        if _CYRILLIC.search(question):
            marker = self.cfg.no_answer_marker
            system = _SYSTEM_RU.format(marker=marker)
        else:
            marker = _EN_MARKER
            system = _SYSTEM_EN.format(marker=marker)
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

        # Абстенция: маркер отказа встречается в ответе (принимаем RU и EN варианты —
        # модель может ответить не тем маркером, который просили).
        norm = text.upper().replace("Ё", "Е")
        markers = {marker, self.cfg.no_answer_marker, _EN_MARKER}
        if any(m.upper().replace("Ё", "Е") in norm for m in markers):
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
