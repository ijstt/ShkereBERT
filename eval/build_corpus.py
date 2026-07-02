"""Подготовка оценочного набора из SQuAD v2.

Чтобы retrieval-оценка была честной и воспроизводимой:
  * берём случайную (сид фиксирован) подвыборку из N вопросов dev-сплита;
  * корпус документов = УНИКАЛЬНЫЕ контексты именно этих вопросов.
Так гарантируется, что gold-контекст каждого вопроса присутствует в корпусе, и при этом
есть дистракторы (контексты других вопросов) — иначе поиск был бы тривиальным.

qrels: для каждого вопроса «золотой» документ — это его исходный контекст (context_id).
"""

from __future__ import annotations

import random

from shkerebert.ingest import Document, load_squad_questions, _hash_id


def build_eval_set(
    split: str = "validation",
    n_questions: int = 2000,
    seed: int = 42,
):
    """Вернуть (documents, questions).

    questions — список dict как в load_squad_questions (+ context_id как qrel).
    documents — уникальные контексты этих вопросов.
    """
    all_q = load_squad_questions(split=split)
    rng = random.Random(seed)
    rng.shuffle(all_q)
    questions = all_q[: min(n_questions, len(all_q))]

    docs: dict[str, Document] = {}
    for q in questions:
        cid = q["context_id"]
        if cid not in docs:
            docs[cid] = Document(
                id=cid,
                text=q["context"],
                title="",
                meta={"source": "squad_v2", "split": split},
            )
    return list(docs.values()), questions


if __name__ == "__main__":
    docs, qs = build_eval_set(n_questions=200)
    n_imp = sum(q["is_impossible"] for q in qs)
    print(f"documents: {len(docs)}  questions: {len(qs)}  "
          f"answerable: {len(qs) - n_imp}  no-answer: {n_imp}")
