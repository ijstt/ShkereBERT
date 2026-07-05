# 05. Загрузка данных (Ingestion)

Модуль `shkerebert/ingest.py` приводит все источники к единому типу `Document`.

## Document — минимальная единица корпуса

```19:24:shkerebert/ingest.py
@dataclass(frozen=True)
class Document:
    id: str
    text: str
    title: str = ""
    meta: dict = field(default_factory=dict)
```

Дальше `chunking.py` режет `Document.text` на `Chunk[]`.

## Генерация ID

```27:29:shkerebert/ingest.py
def _hash_id(text: str, prefix: str = "doc") -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{h}"
```

Детерминированный ID по содержимому — один и тот же текст всегда даёт один id.

## SQuAD v2 (HuggingFace Datasets)

### Документы (контексты)

```33:56:shkerebert/ingest.py
def load_squad_documents(split: str = "validation", max_contexts: int | None = None):
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split=split)
    seen: dict[str, Document] = {}
    for row in ds:
        ctx = row["context"]
        if ctx in seen:
            continue
        seen[ctx] = Document(
            id=_hash_id(ctx, "sq"),
            text=ctx,
            title=row.get("title", ""),
            meta={"source": "squad_v2", "split": split},
        )
```

**Дедупликация контекстов:** в SQuAD один абзац Wikipedia используется многими
вопросами. Для индекса нужен корпус **уникальных** документов, иначе индекс раздувается
дубликатами.

### Вопросы (для оценки)

```59:84:shkerebert/ingest.py
def load_squad_questions(split: str = "validation", max_questions: int | None = None):
    ...
    for row in ds:
        answers = list(row["answers"]["text"])
        out.append({
            "id": row["id"],
            "question": row["question"],
            "context": row["context"],
            "context_id": _hash_id(row["context"], "sq"),
            "title": row.get("title", ""),
            "answers": answers,
            "is_impossible": len(answers) == 0,
        })
```

- **answerable** — `answers` непустой, эталонный span есть в контексте.
- **unanswerable (impossible)** — `answers == []`, в контексте ответа нет.

`context_id` связывает вопрос с документом в индексе — используется в Recall@k.

## Текстовые файлы (.txt)

```88:96:shkerebert/ingest.py
def load_text_file(path: str | Path) -> Document:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    return Document(
        id=_hash_id(text, "txt"),
        text=text,
        title=path.stem,
        meta={"source": "txt", "path": str(path)},
    )
```

`errors="ignore"` — бинарный мусор в «текстовом» файле не роняет загрузку.

## PDF (.pdf)

```99:112:shkerebert/ingest.py
def load_pdf(path: str | Path) -> Document:
    from pypdf import PdfReader
    path = Path(path)
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n\n".join(p for p in pages if p)
    return Document(
        id=_hash_id(text or str(path), "pdf"),
        ...
        meta={"source": "pdf", "path": str(path), "n_pages": len(reader.pages)},
    )
```

**Ограничения pypdf:**

- нет OCR — сканы без текстового слоя дают пустой текст;
- сложная вёрстка (колонки, таблицы) извлекается с артефактами;
- страницы склеиваются через `\n\n`.

## Автовыбор по расширению

```115:120:shkerebert/ingest.py
def load_document(path: str | Path) -> Document:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return load_pdf(path)
    return load_text_file(path)
```

## Демо-документы

| Файл | Язык | Назначение |
|------|------|------------|
| `demo/machine_learning.txt` | EN | Дефолт для `./run.sh chat` |
| `demo/bank_products_ru.txt` | RU | Демо банковских продуктов для `./run.sh ru` |

## Оценочный корпус (eval/build_corpus.py)

Для честной retrieval-оценки корпус строится иначе, чем «весь SQuAD»:

```19:44:eval/build_corpus.py
def build_eval_set(split="validation", n_questions=2000, seed=42):
    all_q = load_squad_questions(split=split)
    rng = random.Random(seed)
    rng.shuffle(all_q)
    questions = all_q[: min(n_questions, len(all_q))]
    docs: dict[str, Document] = {}
    for q in questions:
        cid = q["context_id"]
        if cid not in docs:
            docs[cid] = Document(id=cid, text=q["context"], ...)
    return list(docs.values()), questions
```

**Почему так:**

1. Gold-контекст каждого вопроса **гарантированно** в корпусе.
2. Есть **дистракторы** — контексты других вопросов из той же выборки.
3. Фиксированный `seed=42` → воспроизводимость.

Без дистракторов retrieval был бы тривиальным (один документ = один контекст).
