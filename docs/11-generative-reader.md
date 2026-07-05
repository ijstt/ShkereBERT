# 11. Генеративный ридер (Qwen GGUF)

Модуль `shkerebert/generator.py`.

## Зачем generative поверх extractive

Extractive возвращает сырой span — часто неудобно для пользователя и не работает на
составных ответах. **Generative** LLM формулирует естественный ответ по найденным
фрагментам, с явными цитатами `[1]`, `[2]`.

## llama-cpp-python

Локальный инференс **GGUF**-квантизованных моделей без GPU:

```32:36:shkerebert/generator.py
@lru_cache(maxsize=1)
def _load_llm(model_path, n_ctx, n_threads):
    from llama_cpp import Llama
    return Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)
```

| Параметр | Default | Смысл |
|----------|---------|-------|
| model_path | `models/qwen2.5-3b-instruct-q4_k_m.gguf` | ~2 ГБ Q4_K_M |
| n_ctx | 4096 | Размер контекстного окна |
| n_threads | 0 | Все CPU-ядра |
| temperature | 0.0 | Greedy / deterministic |

**GGUF** — формат llama.cpp с квантизацией (Q4 = 4-bit) для CPU.

**Qwen 2.5 3B Instruct** — instruction-tuned LLM, поддерживает RU и EN.

## Grounded generation — контракт промпта

Модель получает system + user prompt. System задаёт правила:

```43:65:shkerebert/generator.py
_SYSTEM_RU = (
    "Ты — ассистент, который отвечает на вопрос СТРОГО по предоставленным фрагментам..."
    "4. Если ответа в фрагментах НЕТ — ответь ровно: {marker}\n"
    ...
)
_SYSTEM_EN = (
    "You are an assistant that answers the question STRICTLY from the provided document fragments..."
    "4. If the fragments do NOT contain the answer, reply exactly: {marker}\n"
    ...
)
```

**Ключевые правила:**

1. Только информация из фрагментов — **запрет внешних знаний** (иначе галлюцинации на
   unanswerable: модель «знает» Telenet, Galileo из pretraining).
2. Ответ текстом, цитаты `[n]` **в конце**.
3. Маркер отказа: `НЕТ ОТВЕТА` (RU) / `NO ANSWER` (EN).

## Выбор языка промпта

```86:91:shkerebert/generator.py
        if _CYRILLIC.search(question):
            marker = self.cfg.no_answer_marker
            system = _SYSTEM_RU.format(marker=marker)
        else:
            marker = _EN_MARKER
            system = _SYSTEM_EN.format(marker=marker)
```

Русский system-prompt на EN-вопросах провоцировал Qwen отвечать по-русски — исправлено
эвристикой по кириллице (`_CYRILLIC = re.compile(r"[а-яё]", re.I)`).

## User prompt — нумерованные фрагменты

```77:83:shkerebert/generator.py
    def _build_prompt(self, question, chunks):
        for i, ch in enumerate(chunks, 1):
            blocks.append(f"[{i}]{title} {ch.text}")
        return f"Фрагменты:\n{context}\n\nВопрос: {question}"
```

Номера `[1]..[k]` — для цитирования и маппинга на `chunk.id`.

## generate()

```94:119:shkerebert/generator.py
        out = self.llm.create_chat_completion(
            messages=[{"role": "system", ...}, {"role": "user", ...}],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        text = out["choices"][0]["message"]["content"].strip()

        # Абстенция по маркеру (RU и EN)
        if any(m.upper()... in norm for m in markers):
            return GenAnswer("", False, ...)

        idxs = sorted({int(m) for m in re.findall(r"\[(\d+)\]", text) ...})
        cited_ids = [chunks[i - 1].id for i in idxs]
        return GenAnswer(answer=text, is_answerable=True, cited_indices=idxs, ...)
```

## GenAnswer

```22:29:shkerebert/generator.py
@dataclass
class GenAnswer:
    answer: str
    is_answerable: bool
    cited_indices: list[int]      # 1-based
    cited_chunk_ids: list[str]
    raw: str
    latency_s: float
```

## Оценка generative — особенности метрик

SQuAD EM/F1 ждут **короткий span**. LLM отвечает фразой → F1 штрафует многословие.

`eval/eval_generative.py` отчитывает:

- **строгий squad_v2** после `clean_generated()` (снятие `[n]`, префиксов);
- **gold-containment** — нормализованный эталон ⊆ ответ;
- качество абстенции (recall на impossible, false abstain на answerable);
- latency mean/p50/p95.

Типично: containment ~0.89 при strict F1 ~44 на generative.

## On-premise / офлайн

Все данные остаются на машине — документ не уходит во внешний API. Важно для банковского
сценария (`demo/bank_products_ru.txt`).

## Зависимость

```14:15:requirements.txt
llama-cpp-python>=0.3
```

Сборка может требовать компилятор C++ (cmake) при `pip install`.
