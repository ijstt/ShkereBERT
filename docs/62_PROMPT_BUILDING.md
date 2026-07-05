# Построение промпта для генеративного ридера (Qwen)

## Принципы grounded-промпта

1. **Строгое заземление** — «только по фрагментам, никаких внешних знаний»
2. **Обязательные цитаты** — номера фрагментов `[1]`, `[2]` в конце ответа
3. **Маркер отказа** — явная инструкция вернуть специальную строку, если ответа нет
4. **Языковая адаптация** — промпт на языке вопроса (RU/EN)
5. **Запрет пустых скобок** — Qwen иногда отвечал только `[2]` без текста

## System Prompt (RU)

```python
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
```

## System Prompt (EN)

```python
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
```

## Выбор промпта по языку вопроса

```python
_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)

if _CYRILLIC.search(question):
    marker = self.cfg.no_answer_marker  # "НЕТ ОТВЕТА"
    system = _SYSTEM_RU.format(marker=marker)
else:
    marker = "NO ANSWER"
    system = _SYSTEM_EN.format(marker=marker)
```

**Почему это важно**: на EN-вопросах русский промпт заставлял Qwen отвечать по-русски
(найдено в `eval_generative.py`). Языковая адаптация устранила этот баг.

## User Prompt (контекст + вопрос)

```python
def _build_prompt(self, question: str, chunks: list[Chunk]) -> str:
    blocks = []
    for i, ch in enumerate(chunks, 1):
        title = f" ({ch.title})" if ch.title else ""
        blocks.append(f"[{i}]{title} {ch.text}")
    context = "\n\n".join(blocks)
    return f"Фрагменты:\n{context}\n\nВопрос: {question}"
```

- Фрагменты нумеруются **1-based** для удобства цитирования
- Title добавляется в скобках (для SQuAD — название статьи Википедии)
- Разделитель `\n\n` — читаемо для модели

## Параметры генерации

```python
GeneratorConfig(
    temperature=0.0,      # детерминированно
    max_tokens=256,       # достаточно для ответа + цитат
    n_ctx=4096,           # контекстное окно Qwen
    n_threads=0,          # все ядра CPU
)
```

## Постобработка ответа

```python
# 1. Проверка маркера отказа (RU + EN варианты)
norm = text.upper().replace("Ё", "Е")
markers = {marker, self.cfg.no_answer_marker, "NO ANSWER"}
if any(m.upper().replace("Ё", "Е") in norm for m in markers):
    return GenAnswer("", False, raw=text, latency_s=dt)

# 2. Извлечение цитат [n]
idxs = sorted({int(m) for m in re.findall(r"\[(\d+)\]", text)
               if 1 <= int(m) <= len(chunks)})
cited_ids = [chunks[i - 1].id for i in idxs]

return GenAnswer(answer=text, is_answerable=True,
                 cited_indices=idxs, cited_chunk_ids=cited_ids,
                 raw=text, latency_s=dt)
```

## Найденные и исправленные баги промпта

| Баг | Симптом | Фикс |
|---|---|---|
| Пустые скобки | Qwen отвечал `[2]` без текста | Правило 3: «НИКОГДА не отвечай одними скобками» |
| RU-промпт на EN | Ответы по-русски на английские вопросы | Языковая адаптация промпта |
| Галлюцинации на impossible | Qwen отвечал из памяти (Telenet, Galileo) | Правило 1: «Никаких внешних знаний» + UNVERIFIED в hybrid |
| Ё/Е в маркере | «НЕТ ОТВЕТА» vs «НЕТ ОТВЕТА» | Нормализация `replace("Ё", "Е")` при проверке |

## Оценка промпта (eval/eval_generative.py)

Метрики на 150 вопросах (seed=123):
- **Gold-containment**: 0.89 (эталон в ответе в