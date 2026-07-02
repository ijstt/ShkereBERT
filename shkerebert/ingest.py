"""Загрузка документов из разных источников в единый вид `Document`.

Поддерживаем три источника:
  * SQuAD v2 (HuggingFace Datasets) — контексты дедуплицируются в корпус документов;
  * .txt — произвольный текстовый файл;
  * .pdf — извлечение текста через pypdf (для демо на своём документе).

`Document` — минимальная единица: уникальный текст + метаданные. Дальше документ
режется на `Chunk`-и (см. chunking.py).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    title: str = ""
    meta: dict = field(default_factory=dict)


def _hash_id(text: str, prefix: str = "doc") -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{h}"


# --- SQuAD v2 --------------------------------------------------------------
def load_squad_documents(split: str = "validation", max_contexts: int | None = None):
    """Уникальные контексты SQuAD v2 как список `Document`.

    В SQuAD один контекст переиспользуется многими вопросами, поэтому дедуплицируем:
    это и есть «документная база», по которой ретривер ищет. Порядок сохраняется
    (детерминированность), первый встреченный title закрепляется за контекстом.
    """
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
        if max_contexts is not None and len(seen) >= max_contexts:
            break
    return list(seen.values())


def load_squad_questions(split: str = "validation", max_questions: int | None = None):
    """Вопросы SQuAD v2 в удобном для оценки виде.

    Возвращает список словарей: id, question, context, answers (list[str]),
    is_impossible (bool). Unanswerable-вопросы имеют пустой список ответов.
    """
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split=split)
    if max_questions is not None:
        ds = ds.select(range(min(max_questions, len(ds))))
    out = []
    for row in ds:
        answers = list(row["answers"]["text"])
        out.append(
            {
                "id": row["id"],
                "question": row["question"],
                "context": row["context"],
                "context_id": _hash_id(row["context"], "sq"),
                "answers": answers,
                "is_impossible": len(answers) == 0,
            }
        )
    return out


# --- Файлы -----------------------------------------------------------------
def load_text_file(path: str | Path) -> Document:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    return Document(
        id=_hash_id(text, "txt"),
        text=text,
        title=path.stem,
        meta={"source": "txt", "path": str(path)},
    )


def load_pdf(path: str | Path) -> Document:
    """Извлечь текст из PDF постранично (pypdf). Страницы склеиваются через \\n\\n."""
    from pypdf import PdfReader

    path = Path(path)
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n\n".join(p for p in pages if p)
    return Document(
        id=_hash_id(text or str(path), "pdf"),
        text=text,
        title=path.stem,
        meta={"source": "pdf", "path": str(path), "n_pages": len(reader.pages)},
    )


def load_document(path: str | Path) -> Document:
    """Автовыбор загрузчика по расширению файла."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return load_pdf(path)
    return load_text_file(path)
