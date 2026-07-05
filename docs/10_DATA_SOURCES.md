# Источники данных и инжест

## Поддерживаемые форматы

| Формат | Загрузчик | Особенности |
|---|---|---|
| **SQuAD v2** | `load_squad_documents()` | Дедупликация контекстов по тексту; title сохраняется |
| **PDF** | `load_pdf()` → `pypdf.PdfReader` | Только текстовый слой; сканы/таблицы теряются |
| **TXT** | `load_text_file()` | UTF-8, ошибки игнорируются |

## SQuAD v2 как оценочный корпус

```python
from shkerebert.ingest import load_squad_documents, load_squad_questions

# Документы = уникальные контексты validation-сплита
docs = load_squad_documents(split="validation", max_contexts=200)
# → List[Document], len ≈ 150–180 (в SQuAD контексты переиспользуются)

# Вопросы в удобном формате для оценки
questions = load_squad_questions(split="validation", max_questions=500)
# → List[dict]: id, question, context, context_id, answers[], is_impossible
```

**Почему SQuAD v2:**
- Единственный массовый QA-датасет с **намеренно безответными вопросами** (≈33%)
- Официальная метрика `squad_v2` в HF `evaluate` — EM/F1/HasAns/NoAns
- Контексты — абзацы Википедии (энциклопедический стиль, чистый текст)

## Структура Document

```python
@dataclass(frozen=True)
class Document:
    id: str          # sha1(text)[:12] с префиксом "sq"/"txt"/"pdf"
    text: str        # полный текст документа
    title: str       # заголовок (для SQuAD — title статьи)
    meta: dict       # источник, путь, split, n_pages и т.д.
```

## Дедупликация SQuAD

В SQuAD один контекст обслуживает 3–5 вопросов. `load_squad_documents` оставляет
**уникальные контексты** — это и есть корпус для ретривера. Порядок детерминирован
(первый встреченный title закрепляется).

## Добавление своего источника

1. Реализуйте функцию `load_my_format(path) -> Document` в `ingest.py`
2. Добавьте ветку в `load_document(path)` по расширению
3. Для PDF со сканами — подключите OCR (tesseract / paddleocr / unstructured)

## Ограничения текущего инжеста

| Проблема | Статус | План |
|---|---|---|
| Таблицы в PDF теряются | Известно | `pdfplumber` / `unstructured` + table-to-text |
| Сканы без текстового слоя | Не поддерживается | OCR-слой перед индексацией |
| Многостраничные структуры (оглавления, колонтитулы) | Не обрабатываются | Пост-обработка текста |
| Документы > 100k токенов | Работает, но чанкинг критичен | См. `20_CHUNKING_STRATEGY.md` |
