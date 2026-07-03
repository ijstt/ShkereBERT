"""Честный эксперимент с чанкингом: ДЛИННЫЕ документы.

Проблема, найденная в v1: контексты SQuAD короткие (медиана ~141 токен), поэтому при
chunk_size>=256 чанкер почти не срабатывает (256: режется 8% документов, 512: 0.3%) —
и вывод «размер чанка почти не влияет» нельзя переносить на реальные длинные PDF.

Здесь мы устраняем эту дыру: склеиваем все уникальные контексты одной статьи Википедии
(поле title в SQuAD) в один длинный документ (тысячи токенов). Теперь:
  * чанкинг реально режет каждый документ на десятки фрагментов;
  * внутри документа появляются «внутренние дистракторы» — соседние абзацы той же
    статьи, семантически близкие к вопросу (реалистичный режим ошибок retrieval);
  * свип chunk_size измеряет именно то, что заявлен измерять.

Overlap масштабируем пропорционально (25% от size), чтобы менять одну переменную.

Метрики на held-out test (tau калибруется на calibration-сплите, без утечки):
  * answer-recall@k — доля answerable-вопросов, у которых хотя бы один из top-k чанков
    содержит эталонный ответ строкой (ловит и разрезание ответа границей чанка);
  * doc-recall@k — хотя бы один top-k чанк из «золотой» статьи (грубее, для сравнения);
  * EM / F1 / NoAns_F1 (официальная метрика squad_v2) end-to-end;
  * латентность на вопрос.

Результаты -> eval/results/longdoc_chunking.csv + longdoc_chunking.png
"""

from __future__ import annotations

import argparse
import random
from dataclasses import replace

import pandas as pd

from shkerebert.chunking import TokenCounter, chunk_documents
from shkerebert.config import Config, RESULTS_DIR, default_config
from shkerebert.ingest import Document, load_squad_questions, _hash_id
from shkerebert.reader import Reader
from shkerebert.retriever import DenseRetriever
from eval.eval_e2e import calibrate, collect_raw_predictions, score_fixed_tau


def build_longdoc_eval_set(n_questions: int = 600, seed: int = 42, split: str = "validation"):
    """Вернуть (long_documents, questions).

    Документ = конкатенация ВСЕХ уникальных контекстов статьи (title) из сплита —
    включая абзацы, к которым нет вопросов в выборке (естественные дистракторы).
    """
    all_q = load_squad_questions(split=split)
    rng = random.Random(seed)
    sampled = all_q[:]
    rng.shuffle(sampled)
    questions = sampled[: min(n_questions, len(sampled))]
    titles = {q["title"] for q in questions}

    by_title: dict[str, list[str]] = {}
    seen: set[str] = set()
    for q in all_q:  # исходный порядок сплита -> детерминированная сборка
        if q["title"] in titles and q["context"] not in seen:
            seen.add(q["context"])
            by_title.setdefault(q["title"], []).append(q["context"])

    docs = [
        Document(
            id=_hash_id(title, "long"),
            text="\n\n".join(ctxs),
            title=title,
            meta={"source": "squad_v2_longdoc", "n_paragraphs": len(ctxs)},
        )
        for title, ctxs in by_title.items()
    ]
    return docs, questions


def corpus_stats(docs, cfg: Config) -> dict:
    counter = TokenCounter(cfg.retriever.embed_model)
    lengths = [counter.count(d.text) for d in docs]
    s = pd.Series(lengths)
    return {
        "n_docs": len(docs),
        "doc_tokens_median": int(s.median()),
        "doc_tokens_min": int(s.min()),
        "doc_tokens_max": int(s.max()),
    }


def answer_recall(questions, retriever: DenseRetriever, k: int):
    """(answer-recall@k, doc-recall@k) по answerable-вопросам, батч-поиском."""
    qs = [q for q in questions if not q["is_impossible"]]
    qvecs = retriever.embedder.encode([q["question"] for q in qs])
    all_hits = retriever.index.search(qvecs, k=k)
    n_ans = n_doc = 0
    for q, hits in zip(qs, all_hits):
        texts = [h.chunk.text.lower() for h in hits]
        if any(any(a.lower() in t for t in texts) for a in q["answers"]):
            n_ans += 1
        if any(h.chunk.title == q["title"] for h in hits):
            n_doc += 1
    return n_ans / max(1, len(qs)), n_doc / max(1, len(qs)), len(qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="всего вопросов (делятся 50/50)")
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--calib-frac", type=float, default=0.5)
    args = ap.parse_args()

    base = default_config()
    base = replace(base, retriever=replace(base.retriever, top_k=args.top_k))
    docs, questions = build_longdoc_eval_set(n_questions=args.n, seed=base.seed)
    n_cal = int(len(questions) * args.calib_frac)
    calib_q, test_q = questions[:n_cal], questions[n_cal:]

    stats = corpus_stats(docs, base)
    print(f"Длинные документы: {stats} | calibration={len(calib_q)} test={len(test_q)}")

    reader = Reader(base.reader)
    rows = []
    for size in args.chunk_sizes:
        cfg = replace(base, chunk=replace(base.chunk, size=size, overlap=size // 4))
        chunks = chunk_documents(docs, cfg.chunk, cfg.retriever.embed_model)
        n_multi = len({c.doc_id for c in chunks if c.index > 0})
        print(f"[chunk={size} overlap={size // 4}] чанков={len(chunks)}, "
              f"докам >1 чанка: {n_multi}/{len(docs)}. Строю индекс...")
        retriever = DenseRetriever.build(docs, cfg, show_progress=False)

        raw_cal, _ = collect_raw_predictions(cfg, calib_q, retriever, reader)
        raw_test, lat = collect_raw_predictions(cfg, test_q, retriever, reader)
        _, best = calibrate(cfg, raw_cal)
        tau = float(best["tau"])
        res = score_fixed_tau(cfg, raw_test, tau)
        rec_ans, rec_doc, n_answerable = answer_recall(test_q, retriever, args.top_k)

        row = {
            "chunk_size": size,
            "overlap": size // 4,
            "n_chunks": len(chunks),
            "docs_split": f"{n_multi}/{len(docs)}",
            "answer_recall@k": round(rec_ans, 4),
            "doc_recall@k": round(rec_doc, 4),
            "EM": round(res["exact"], 2),
            "F1": round(res["f1"], 2),
            "HasAns_F1": round(res.get("HasAns_f1", float("nan")), 2),
            "NoAns_F1": round(res.get("NoAns_f1", float("nan")), 2),
            "tau*": round(tau, 3),
            "latency_ms/q": round(lat, 1),
        }
        print("  ", row)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["top_k"] = args.top_k
    df["n_test"] = len(test_q)
    df["n_answerable_test"] = n_answerable
    for k, v in stats.items():
        df[k] = v
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "longdoc_chunking.csv", index=False)

    _plot(df, args.top_k)
    print("\n", df.to_string(index=False))
    print(f"\nСохранено: {RESULTS_DIR / 'longdoc_chunking.csv'} и longdoc_chunking.png")


def _plot(df: pd.DataFrame, k: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(df["chunk_size"], df["F1"], marker="o", lw=2, label="overall F1")
    ax1.plot(df["chunk_size"], df["HasAns_F1"], marker="s", ls="--", label="HasAns F1")
    ax1.set_xlabel("chunk_size (токены)")
    ax1.set_ylabel("F1 (test)")
    ax1.set_title("Длинные документы: F1 vs размер чанка")
    ax1.legend()

    ax2.plot(df["chunk_size"], df["answer_recall@k"], marker="o", lw=2,
             label=f"answer-recall@{k}")
    ax2.plot(df["chunk_size"], df["doc_recall@k"], marker="s", ls="--",
             label=f"doc-recall@{k}")
    ax2.set_xlabel("chunk_size (токены)")
    ax2.set_ylabel("recall")
    ax2.set_title("Длинные документы: retrieval vs размер чанка")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "longdoc_chunking.png", dpi=130)


if __name__ == "__main__":
    main()
